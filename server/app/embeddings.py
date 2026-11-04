"""中文向量模型:BGE-small-zh-v1.5,两种运行模式(路线二 第五步)。

- EMBEDDING_MODE=local(默认):fastembed 进程内加载 ONNX,CPU 轻量运行,
  模型缓存到 DATA_DIR/models(首次构建索引时自动下载,约 50MB);
- EMBEDDING_MODE=tei:调用独立部署的 text-embeddings-inference 服务(TEI_URL),
  向量计算从应用实例剥离,多实例横向扩容时不再每实例常驻一份模型。

BGE 检索式模型对查询加指令前缀以提升召回质量;两种模式共用此前缀,向量空间一致。
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 部分网络环境无法访问 hf-xet 存储端点,禁用后走普通 HTTP 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastembed import TextEmbedding

from . import config

BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章:"


class Embedder:
    _instance: "Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._model: TextEmbedding | None = None
        self._model_lock = threading.Lock()

    @classmethod
    def get(cls) -> "Embedder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = Embedder()
            return cls._instance

    @property
    def _tei(self) -> bool:
        return config.EMBEDDING_MODE == "tei" and bool(config.TEI_URL)

    def _ensure(self) -> TextEmbedding:
        with self._model_lock:
            if self._model is None:
                self._model = TextEmbedding(
                    config.load_settings()["embedding_model"],
                    cache_dir=str(config.MODEL_CACHE),
                )
            return self._model

    @property
    def dim(self) -> int:
        return 512  # bge-small-zh-v1.5

    @property
    def ready(self) -> bool:
        return self._tei or self._model is not None

    # ------------------------------------------------------------ TEI 模式

    def _embed_tei(self, texts: list[str]) -> list[list[float]]:
        """TEI /embed 接口:POST {"inputs": [...]} -> [[...], ...]。
        失败抛出由调用方按分支隔离处理(检索各分支已有 try/except)。"""
        req = urllib.request.Request(
            config.TEI_URL.rstrip("/") + "/embed",
            data=json.dumps({"inputs": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            vectors = json.loads(resp.read().decode("utf-8"))
        return [[float(x) for x in v] for v in vectors]

    # ------------------------------------------------------------ 编码入口

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if self._tei:
            out: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                out.extend(self._embed_tei(texts[i:i + batch_size]))
            return out
        model = self._ensure()
        return [list(map(float, v)) for v in model.embed(texts, batch_size=batch_size)]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([BGE_QUERY_PREFIX + query])[0]

    def embed_plain(self, text: str) -> list[float]:
        """对称相似度场景(如语义路由)使用,不加 BGE 检索指令前缀。"""
        return self.embed_texts([text])[0]

    def warmup(self):
        if self._tei:
            self._embed_tei(["warmup"])   # 连通性自检
        else:
            self._ensure()
