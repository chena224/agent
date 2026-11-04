"""索引构建任务(路线二 第二步)。

- standalone:后台线程执行(与 v5 行为一致),进度写 cache.py(进程内);
- cluster:Celery 任务提交到 Redis 队列,由独立 worker 进程执行;
  进度写 Redis,任意后端实例都能查询;分布式锁保证同一知识库多实例下
  只有一个构建在跑。

核心构建逻辑 execute_build() 两种形态共用,不随调度方式分叉。
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
import time
from pathlib import Path

from . import analytics, cache, config, db
from .chunking import run_strategy
from .embeddings import Embedder
from .parser import apply_captions, ensure_kb_document, load_parsed
from .store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- Celery 实例

celery_app = None
if config.is_cluster():
    from celery import Celery
    celery_app = Celery("rag",
                        broker=config.CELERY_BROKER_URL,
                        backend=config.CELERY_RESULT_BACKEND)
    celery_app.conf.update(
        task_track_started=True,
        task_time_limit=3600,
        worker_prefetch_multiplier=1,   # 构建为重任务,避免单 worker 预取堆积
    )


# ---------------------------------------------------------------- 切分缓存(从 main 迁入,任务与预览共用)

def _chunk_cache_path(kb_id: int) -> Path:
    return config.kb_dir(kb_id) / "chunk_cache.pkl"


def chunk_cache_load(kb_id: int) -> dict:
    try:
        return pickle.loads(_chunk_cache_path(kb_id).read_bytes())
    except Exception:
        return {}


def chunk_cache_save(kb_id: int, key: str, chunks) -> None:
    data = chunk_cache_load(kb_id)
    data[key] = chunks
    try:
        _chunk_cache_path(kb_id).write_bytes(pickle.dumps(data))
    except Exception:
        pass


def chunk_cache_key(strategy: str, params: dict, cap_sig: str = "") -> str:
    # cap_sig:图片描述内容指纹——描述生成/清空后旧切分缓存必须失效
    return strategy + ":" + json.dumps(params, sort_keys=True) + ":" + cap_sig


def caption_signature(captions: dict) -> str:
    if not captions:
        return ""
    raw = json.dumps(sorted(captions.items()), ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def doc_for_chunking(kb_id: int, doc, generate: bool = False,
                     progress_cb=None) -> tuple:
    """返回 (应用了图片描述的 ParsedDoc, 描述指纹, 备注)。
    generate=True 且库开关开启时,为缺描述的图片调视觉模型补齐(仅构建路径);
    预览路径 generate=False:只应用已有缓存,不产生任何 LLM 调用。
    开关关闭时返回裸标记版本(图片位置保留、无描述)——开关语义严格。
    """
    from . import vision
    rc = (db.get_kb(kb_id) or {}).get("retrieval") or {}
    note = ""
    ocr_n = len(getattr(doc, "ocr_pages", []) or [])
    if ocr_n:
        note = f"检测到扫描页 {ocr_n} 页,已 OCR 转录"
    if not doc.images:
        return apply_captions(doc, {}), "", note
    if not rc.get("image_caption", True):
        return apply_captions(doc, {}), "", note
    captions = vision.load_captions(kb_id)
    if generate:
        captions, note = vision.caption_images(kb_id, doc, progress_cb=progress_cb)
        if ocr_n:
            note = (note + ";" if note else "") + f"检测到扫描页 {ocr_n} 页,已 OCR 转录"
    missing = len(doc.images) - sum(1 for im in doc.images if captions.get(im["img_id"]))
    if not note and missing:
        note = f"{missing} 张图片暂无描述(构建索引时生成)"
    return apply_captions(doc, captions), caption_signature(captions), note


# ---------------------------------------------------------------- 构建核心

def execute_build(kb_id: int, strategy: str, params: dict,
                  user_id: int, username: str, doc_id: str = "") -> None:
    """执行索引构建;进度写 cache.progress(standalone=进程内,cluster=Redis)。
    异常不抛出——进度 stage=error 即任务终态,由前端轮询感知。"""
    kb = db.get_kb(kb_id)
    kb_name = kb["name"] if kb else str(kb_id)
    prog = lambda **kw: cache.progress_set(f"build:{kb_id}", **kw)
    t0 = time.time()
    try:
        doc = load_parsed(config.kb_dir(kb_id) / "document.json") or ensure_kb_document(kb_id)
        if doc is None:
            raise RuntimeError("该知识库尚未加载文档,请先上传文档")
        # 图片描述(视觉模型):仅构建路径生成;按库开关 image_caption 控制
        note = ""
        cap_sig = ""
        rc = (kb or {}).get("retrieval") or {}
        img_on = bool(doc.images) and bool(rc.get("image_caption", True))
        if img_on:
            def _cap_cb(i: int, n: int) -> None:
                prog(stage="caption", percent=4 + int(4 * i / max(1, n)),
                     message=f"图片描述生成中 {i}/{n}(视觉模型)")
            prog(stage="caption", percent=4, message="图片描述生成中…")
        if doc.images:
            doc, cap_sig, note = doc_for_chunking(
                kb_id, doc, generate=img_on,
                progress_cb=_cap_cb if img_on else None)
        prog(stage="chunking", percent=10,
             message=f"文档切分中…{('(' + note + ')') if note else ''}")
        key = chunk_cache_key(strategy, params, cap_sig)
        chunks = chunk_cache_load(kb_id).get(key)
        if chunks is None:
            embedder = Embedder.get() if strategy == "semantic" else None
            chunks = run_strategy(strategy, doc, embedder=embedder, params=params)
            chunk_cache_save(kb_id, key, chunks)

        prog(stage="embedding", percent=18,
             message=f"向量化 {len(chunks)} 个分块(首次需下载模型)…")
        emb = Embedder.get()
        vectors = []
        batch = 48
        for i in range(0, len(chunks), batch):
            vectors.extend(emb.embed_texts([c.text for c in chunks[i:i + batch]]))
            pct = 18 + int(67 * min(1.0, (i + batch) / len(chunks)))
            prog(stage="embedding", percent=pct,
                 message=f"向量化中 {min(i + batch, len(chunks))}/{len(chunks)}")

        prog(stage="inserting", percent=90, message="写入 Milvus…")
        info = VectorStore.get().build(kb_id, chunks, vectors, strategy,
                                       doc.name, emb.dim, doc_id=doc_id)
        # 业务库与向量库双写:先向量后台账,失败有进度可查;一致性核对脚本兜底
        db.update_kb(kb_id, {"doc_name": doc.name, "strategy": strategy,
                             "chunk_count": info["count"], "built_at": info["built_at"]})
        if not doc_id:   # 未登记台账的旧流程文档,构建时补登(幂等:按内容指纹)
            db.add_document(kb_id, doc.name, doc.page_count, len(doc.full_text),
                            len(doc.tables), username)
        analytics.record("build", strategy=strategy,
                         latency_ms=int((time.time() - t0) * 1000),
                         sources_n=info["count"], kb_id=kb_id, kb_name=kb_name,
                         user_id=user_id, username=username)
        prog(stage="done", percent=100, message=f"完成:{info['count']} 个分块")
    except Exception as e:
        logger.exception("索引构建失败 kb=%s", kb_id)
        prog(stage="error", percent=0, message=str(e)[:300])


if celery_app is not None:
    @celery_app.task(name="rag.build_index", bind=True, max_retries=0)
    def build_index_task(self, kb_id: int, strategy: str, params: dict,
                         user_id: int, username: str, doc_id: str = ""):
        """Celery 构建任务。分布式锁兜底:即使同一 kb 被重复提交也只跑一个。"""
        with cache.build_lock(f"build_kb_{kb_id}", ttl=3600) as ok:
            if not ok:
                cache.progress_set(f"build:{kb_id}", stage="error", percent=0,
                                   message="已有构建任务在执行,本次提交被忽略")
                return
            execute_build(kb_id, strategy, params, user_id, username, doc_id)


# ---------------------------------------------------------------- 提交入口

def submit_build(kb_id: int, strategy: str, params: dict,
                 user_id: int, username: str, doc_id: str = "") -> str:
    """提交构建任务;返回任务标识(cluster=celery task_id,standalone='thread')。
    未抢到构建锁时抛 RuntimeError(上层转 409)。"""
    if celery_app is not None:
        # 先抢锁再入队,避免队列里堆积同一 kb 的重复任务
        with cache.build_lock(f"submit_kb_{kb_id}", ttl=30) as ok:
            if not ok:
                raise RuntimeError("该知识库索引正在构建中,请稍候")
            build_index_task.delay(kb_id, strategy, params, user_id, username, doc_id)
        return "celery"

    # standalone:线程路径。与 cluster 同一把 build_kb_ 锁(内存后端即非阻塞
    # threading.Lock),提交时原子抢占、持有至构建结束——locked()+acquire 分离
    # 的写法有 TOCTOU 窗口,两个并发请求都会返回 ok,但一个其实在干等。
    lock = cache.build_lock(f"build_kb_{kb_id}", ttl=3600)
    if not lock.__enter__():
        raise RuntimeError("该知识库索引正在构建中,请稍候")

    def work():
        try:
            execute_build(kb_id, strategy, params, user_id, username, doc_id)
        finally:
            lock.__exit__(None, None, None)

    threading.Thread(target=work, daemon=True).start()
    return "thread"


def build_progress(kb_id: int) -> dict:
    return cache.progress_get(f"build:{kb_id}") or \
        {"stage": "idle", "percent": 0, "message": ""}
