"""向量库:Milvus(路线二 第四步)。

- standalone:MILVUS_URI 未配置时用 Milvus Lite,数据落盘 DATA_DIR/milvus_lite.db;
- cluster:MILVUS_URI=http://host:19530 接独立部署的 Milvus,多实例共享向量数据。
两种形态同一套 MilvusClient API,仅 URI 不同;向量只放 Milvus,业务数据在关系库。

多知识库架构:每个知识库一个独立集合 kb_{kb_id},实现知识库间的数据隔离。
集合 schema:
  id      INT64 自增主键
  vector  FLOAT_VECTOR(dim)
  text    VARCHAR  块正文(表格为 markdown;父子块策略下为子块)
  kind    VARCHAR  text | table
  doc_id  VARCHAR  文档台账 ID(路线二:文档级删除/双库一致性核对的锚点;旧数据为 "")
  meta    VARCHAR  JSON 字符串(页码/章节/条目/策略/父块等)
"""
from __future__ import annotations

import json
import threading
import time

from pymilvus import MilvusClient, DataType

from . import config
from .chunking import Chunk

MAX_TEXT = 8000
MAX_META = 6000


def collection_name(kb_id: int) -> str:
    return f"kb_{kb_id}"


def _meta_file(kb_id: int):
    return config.kb_dir(kb_id) / "index_meta.json"


class VectorStore:
    _instance: "VectorStore | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._client: MilvusClient | None = None
        self._client_lock = threading.Lock()

    @classmethod
    def get(cls) -> "VectorStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = VectorStore()
            return cls._instance

    @property
    def client(self) -> MilvusClient:
        with self._client_lock:
            if self._client is None:
                uri = config.MILVUS_URI or str(config.MILVUS_FILE)
                # 独立 Milvus 服务启动/故障转移时连接可能短暂失败,做有限重试;
                # Lite(本地文件)一次即成功,重试无副作用
                last: Exception | None = None
                for attempt in range(3):
                    try:
                        self._client = MilvusClient(uri, timeout=30)
                        last = None
                        break
                    except Exception as e:
                        last = e
                        time.sleep(min(2 ** attempt, 4))
                if last is not None:
                    raise last
            return self._client

    # ------------------------------------------------------------ 索引构建

    def build(self, kb_id: int, chunks: list[Chunk], vectors: list[list[float]],
              strategy: str, doc_name: str, dim: int, doc_id: str = "") -> dict:
        client = self.client
        coll = collection_name(kb_id)
        if client.has_collection(coll):
            client.drop_collection(coll)
        schema = client.create_schema(auto_id=True, enable_dynamic_fields=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT)
        schema.add_field("kind", DataType.VARCHAR, max_length=16)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("meta", DataType.VARCHAR, max_length=MAX_META)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX",
                               metric_type="COSINE")
        client.create_collection(coll, schema=schema, index_params=index_params)

        rows = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            meta = {
                "page_start": c.page_start, "page_end": c.page_end,
                "chapter": c.chapter, "article": c.article,
                "chunk_id": i, **c.meta,
            }
            rows.append({
                "vector": v,
                "text": c.text[:MAX_TEXT - 100],
                "kind": c.kind,
                "doc_id": doc_id,
                "meta": json.dumps(meta, ensure_ascii=False)[:MAX_META - 100],
            })
        batch = 128
        for i in range(0, len(rows), batch):
            client.insert(coll, rows[i:i + batch])
        client.flush(coll)

        info = {
            "doc_name": doc_name,
            "strategy": strategy,
            "count": len(rows),
            "dim": dim,
            "built_at": int(time.time()),
        }
        _meta_file(kb_id).write_text(json.dumps(info, ensure_ascii=False), "utf-8")
        # 同步构建 BM25 全文索引(混合检索用),通过 meta.chunk_id 与向量集合对齐
        from .retrieval import BM25Index
        BM25Index.build(kb_id, chunks)
        return info

    # ------------------------------------------------------------ 检索

    def search(self, kb_id: int, vector: list[float], top_k: int = 6) -> list[dict]:
        client = self.client
        coll = collection_name(kb_id)
        if not client.has_collection(coll):
            return []
        try:
            client.load_collection(coll)
        except Exception:
            pass
        res = client.search(
            coll, data=[vector], limit=top_k,
            output_fields=["text", "kind", "meta"],
        )
        hits: list[dict] = []
        for hit in (res[0] if res else []):
            ent = hit.get("entity", {})
            meta = {}
            try:
                meta = json.loads(ent.get("meta", "{}"))
            except Exception:
                pass
            hits.append({
                "score": round(float(hit.get("distance", 0.0)), 4),
                "text": ent.get("text", ""),
                "kind": ent.get("kind", "text"),
                "meta": meta,
            })
        return hits

    # ------------------------------------------------------------ 状态

    def status(self, kb_id: int) -> dict:
        mf = _meta_file(kb_id)
        if not mf.exists():
            return {"indexed": False, "count": 0}
        try:
            info = json.loads(mf.read_text("utf-8"))
        except Exception:
            return {"indexed": False, "count": 0}
        try:
            coll = collection_name(kb_id)
            if self.client.has_collection(coll):
                try:
                    self.client.load_collection(coll)
                except Exception:
                    pass
                info["indexed"] = True
                return info
        except Exception:
            pass
        return {"indexed": False, "count": 0}

    def clear(self, kb_id: int):
        client = self.client
        coll = collection_name(kb_id)
        if client.has_collection(coll):
            client.drop_collection(coll)
        mf = _meta_file(kb_id)
        if mf.exists():
            mf.unlink()
        from .retrieval import BM25Index
        BM25Index.drop(kb_id)

    # ------------------------------------------------------------ 文档级操作(路线二)

    def delete_by_doc(self, kb_id: int, doc_id: str) -> int:
        """按文档台账 ID 过滤删除向量(双库一致性的 Milvus 侧操作)。返回删除条数。"""
        client = self.client
        coll = collection_name(kb_id)
        if not client.has_collection(coll):
            return 0
        res = client.delete(coll, filter=f'doc_id == "{doc_id}"')
        client.flush(coll)
        # 返回值随版本不同:list[主键] | {"delete_count": n} | int,统一兜底
        if isinstance(res, dict):
            return int(res.get("delete_count", 0))
        if isinstance(res, (list, tuple)):
            return len(res)
        return int(res or 0)

    def doc_chunk_counts(self, kb_id: int) -> dict[str, int]:
        """按 doc_id 聚合当前向量条数,供双库一致性检查脚本与 documents 台账对账。"""
        client = self.client
        coll = collection_name(kb_id)
        if not client.has_collection(coll):
            return {}
        counts: dict[str, int] = {}
        try:
            it = client.query(coll, filter="id >= 0",
                              output_fields=["doc_id"], limit=16384)
            for row in it:
                d = row.get("doc_id", "")
                counts[d] = counts.get(d, 0) + 1
        except Exception:
            pass
        return counts

    def drop_kb(self, kb_id: int):
        """删除知识库时清理其集合与数据目录内容。"""
        self.clear(kb_id)
