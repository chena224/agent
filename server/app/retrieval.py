"""混合检索:BM25(全文) + 向量(语义) 双路召回 + RRF 融合 + 可选精排。

参考业界高 star RAG 项目(RAGFlow / Dify / FastGPT / Langchain-Chatchat / kotaemon)
的通用做法:

- 双路召回:向量检索捕捉语义,BM25 捕捉专有名词/精确词项,互补盲区。
- RRF 融合(Reciprocal Rank Fusion):score = Σ w_branch / (k + rank),
  不需要对两路得分做量纲归一化,比线性加权更稳健(k 默认 60)。
- 检索模式三选一(对齐 Dify):semantic 语义 | fulltext 全文 | hybrid 混合。
- Score 阈值:过滤低相关向量命中;最佳命中余弦低于告警线时标记 low_relevance,
  由上层提示用户"检索相关度较低"(kotaemon 风格)。
- 可选精排:BGE-Reranker(ONNX 本地推理)对融合后候选重排,模型缺失时静默回退。

BM25 索引与 Milvus 集合通过 meta.chunk_id 对齐(建库时枚举序号),
持久化在 kb_dir(kb_id)/bm25.pkl,随索引构建/清除同生共死。
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import threading

from . import config

logger = logging.getLogger(__name__)

RRF_K = 60                  # RRF 平滑常数
WARN_COSINE = 0.45          # 最佳命中余弦低于此值 → 低相关告警
DEFAULT_RETRIEVAL = {
    "mode": "hybrid",          # semantic | fulltext | hybrid
    "top_k": 6,
    "score_threshold": 0.3,    # 向量分支余弦过滤阈值
    "bm25_weight": 0.4,        # 混合模式下 BM25 分支权重(向量 = 1-w)
    "query_rewrite": True,     # 多轮对话时先改写为独立问题
    "rerank": False,           # 是否启用 BGE 精排(需模型文件)
}

_TOKEN_RE = re.compile(r"[\u4e00-\u9fd5]|[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """中文用 jieba 切词,英文/数字按词保留,过滤标点与单字符停用噪声。"""
    import jieba
    out = []
    for tok in jieba.lcut(text.lower()):
        tok = tok.strip()
        if not tok:
            continue
        if _TOKEN_RE.fullmatch(tok) is None and len(tok) <= 1:
            continue
        if tok in ("的", "了", "是", "在", "和", "与", "或", "及", "等", "对", "为"):
            continue
        out.append(tok)
    return out


# ---------------------------------------------------------------- BM25 索引


class BM25Index:
    """单知识库级 BM25 全文索引(rank_bm25,纯 CPU)。"""

    _cache: dict[int, "BM25Index | None"] = {}
    _lock = threading.Lock()

    def __init__(self, records: list[dict]):
        from rank_bm25 import BM25Okapi
        self.records = records                       # [{chunk_id,text,kind,meta}]
        self._tokens = [r["tokens"] for r in records]
        self._bm25 = BM25Okapi(self._tokens) if records else None
        self._by_cid = {r["chunk_id"]: r for r in records}

    # ---------------- 持久化
    @staticmethod
    def _file(kb_id: int):
        return config.kb_dir(kb_id) / "bm25.pkl"

    @classmethod
    def build_records(cls, kb_id: int, records: list[dict]) -> None:
        """records 至少含 chunk_id/text/kind/meta;本函数补齐 tokens 并落盘。"""
        for r in records:
            r["tokens"] = _tokenize(r["text"])
        with open(cls._file(kb_id), "wb") as f:
            pickle.dump(records, f)
        with cls._lock:
            cls._cache[kb_id] = None          # 失效,下次惰性重建

    @classmethod
    def build(cls, kb_id: int, chunks) -> None:
        cls.build_records(kb_id, [{
            "chunk_id": i,
            "text": c.text,
            "kind": c.kind,
            "meta": {"page_start": c.page_start, "page_end": c.page_end,
                     "chapter": c.chapter, "article": c.article, **c.meta},
        } for i, c in enumerate(chunks)])

    @classmethod
    def drop(cls, kb_id: int) -> None:
        try:
            cls._file(kb_id).unlink(missing_ok=True)
        except Exception:
            pass
        with cls._lock:
            cls._cache.pop(kb_id, None)

    @classmethod
    def get(cls, kb_id: int) -> "BM25Index | None":
        with cls._lock:
            if kb_id in cls._cache:
                return cls._cache[kb_id]
        idx = None
        f = cls._file(kb_id)
        if f.exists():
            try:
                with open(f, "rb") as fp:
                    idx = BM25Index(pickle.load(fp))
            except Exception:
                idx = None
        with cls._lock:
            cls._cache[kb_id] = idx
        return idx

    # ---------------- 检索
    def search(self, query: str, top_k: int) -> list[dict]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order[:top_k]:
            if scores[i] <= 0:
                continue
            r = self.records[i]
            hits.append({"score": round(float(scores[i]), 4),
                         "text": r["text"], "kind": r["kind"], "meta": r["meta"]})
        return hits


# ---------------------------------------------------------------- 融合检索


def hybrid_search(kb_id: int, query: str, query_vec: list[float] | None,
                  cfg: dict, store, boost_keywords: list[str] | None = None
                  ) -> tuple[list[dict], dict]:
    """按检索配置执行 语义/全文/混合 检索,返回 (hits, debug)。

    boost_keywords:命中意图携带的增强关键词(语义路由),追加进 BM25 分支
    查询做词项扩展——不同意图走不同检索关键词。
    """
    mode = cfg.get("mode", "hybrid")
    top_k = int(cfg.get("top_k", 6))
    threshold = float(cfg.get("score_threshold", 0.0))
    w_bm25 = float(cfg.get("bm25_weight", 0.4))
    w_vec = 1.0 - w_bm25
    cand = max(top_k * 2, 10)                 # 每路候选数(Chatchat 建议 2~3 倍)

    vec_hits: list[dict] = []
    bm25_hits: list[dict] = []
    vec_err = bm25_err = None
    if mode in ("semantic", "hybrid") and query_vec is not None:
        try:
            vec_hits = store.search(kb_id, query_vec, top_k=cand)
            if threshold > 0:                 # 阈值门控:过滤低相关向量命中
                vec_hits = [h for h in vec_hits if h["score"] >= threshold]
        except Exception as e:                # 向量分支故障不应拖垮全文分支
            vec_err = f"{type(e).__name__}: {e}"
            logger.error("向量分支检索失败 kb=%s: %s", kb_id, vec_err)
    if mode in ("fulltext", "hybrid"):
        try:
            idx = BM25Index.get(kb_id)
            bm25_q = (query if not boost_keywords
                      else query + " " + " ".join(boost_keywords))
            bm25_hits = idx.search(bm25_q, top_k=cand) if idx else []
        except Exception as e:                # BM25 分支故障不应拖垮向量分支
            bm25_err = f"{type(e).__name__}: {e}"
            logger.error("BM25 分支检索失败 kb=%s: %s", kb_id, bm25_err)

    def cid(h: dict):
        return h.get("meta", {}).get("chunk_id", -1)

    if mode == "semantic":
        fused = [{**h, "rrf_score": h["score"], "from": "vector"} for h in vec_hits]
    elif mode == "fulltext":
        fused = [{**h, "rrf_score": h["score"], "from": "bm25"} for h in bm25_hits]
    else:
        # RRF 融合:Σ w / (k + rank),按 chunk_id 去重
        agg: dict = {}
        for branch, hits, w in (("vector", vec_hits, w_vec), ("bm25", bm25_hits, w_bm25)):
            for rank, h in enumerate(hits):
                key = cid(h)
                e = agg.setdefault(key, {"hit": h, "rrf": 0.0, "from": set()})
                e["rrf"] += w / (RRF_K + rank + 1)
                e["from"].add(branch)
                if branch == "vector":        # 保留向量余弦分,用于阈值/告警
                    e["hit"] = h
        fused = [{**e["hit"], "rrf_score": round(e["rrf"], 5),
                  "from": "+".join(sorted(e["from"]))}
                 for e in agg.values()]
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    keep = top_k + 14 if cfg.get("rerank") else top_k   # 启用精排时多留候选
    fused = fused[:keep]

    # 低相关告警(kotaemon):最佳向量命中余弦仍低于告警线(纯全文模式无向量分,不告警)
    best_cosine = max((h["score"] for h in vec_hits), default=0.0)
    low_relevance = (mode != "fulltext" and bool(fused) and best_cosine < WARN_COSINE)

    debug = {
        "mode": mode, "top_k": top_k, "threshold": threshold,
        "vector_hits": len(vec_hits), "bm25_hits": len(bm25_hits),
        "fused": len(fused), "best_cosine": round(best_cosine, 4),
        "low_relevance": low_relevance,
        "boost_keywords": boost_keywords or [],
    }
    # 分支级错误上报(告警用):某一分支挂了时另一分支仍可用,但运维侧应可见
    if vec_err:
        debug["vector_error"] = vec_err
    if bm25_err:
        debug["bm25_error"] = bm25_err
    return fused[:top_k], debug


# ---------------------------------------------------------------- 可选精排


class Reranker:
    """BGE-Reranker(ONNX,fastembed 本地推理)。模型缺失时静默回退,不影响主链路。"""

    _instance: "Reranker | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._tried = False
        self._load_error: str | None = None   # 模型不可用的原因(持久,供诊断)
        self.last_error: str | None = None    # 本次调用的错误(每次 rerank 入口清空)

    @classmethod
    def get(cls) -> "Reranker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = Reranker()
            return cls._instance

    def available(self) -> bool:
        if os.environ.get("RERANK_ENABLED", "1") == "0":
            return False
        if self._tried:
            return self._model is not None
        self._tried = True
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._model = TextCrossEncoder("BAAI/bge-reranker-base",
                                           cache_dir=str(config.MODEL_CACHE))
            self._load_error = None
        except Exception as e:
            self._model = None
            self._load_error = f"{type(e).__name__}: {e}"
            logger.warning("精排模型加载失败,回退 RRF 排序: %s", self._load_error)
        return self._model is not None

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        self.last_error = None               # 单次调用级状态,避免残留误报
        if not self.available() or not hits:
            if not self.available() and self._load_error:
                self.last_error = self._load_error
            return hits
        try:
            scores = list(self._model.rerank(query, [h["text"] for h in hits]))
            for h, s in zip(hits, scores):
                h["rerank_score"] = round(float(s), 4)
            hits.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            logger.error("精排推理失败,保留 RRF 顺序: %s", self.last_error)
        return hits[:top_k]
