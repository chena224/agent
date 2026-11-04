"""语义路由:参考 aurelio-labs/semantic-router 的 Route/utterances 模式。

三级漏斗(自上而下,命中即短路):
  rule_router(关键词规则) → semantic_router(示例句向量) → LLM 分类兜底(agent.py)

- 每个意图(Route)= 名称 + 示例句(utterances) + 检索增强关键词。
- 示例句在首次路由时嵌入为向量并缓存;用户问题嵌入后与全部示例句做余弦
  相似度,取每意图的最高分;最高分过阈值即命中,全程 0 次 LLM 调用。
- 低于阈值返回 None,由上层回退到 LLM 分类器(两级智能路由)。
- 命中意图携带 boost_keywords,检索阶段注入 BM25 分支做查询扩展——
  不同意图走不同的检索关键词(对应 multi-domain routing 思路)。
- 内置全局路由(kb_id=0):chitchat / out_of_scope;知识库级路由均为
  kb_query 的子类型(命中后照常检索生成,仅意图标签与检索关键词不同)。

设计取舍(重要):
- 语义路由只看用户最后一句话,不看对话历史。多轮远距离指代(如隔多轮的
  "那处分呢")在语义路由层大概率低置信 → 自动落到 LLM 兜底分类(该层能看到
  trim 后的完整对话上下文)。这是刻意分工:语义路由追求 0 延迟的确定性命中,
  模糊/指代场景交给有上下文的 LLM,而非把历史拼进路由拖慢每一轮。
- 缓存为 LRU(TTL 兜底):路由配置经 CRUD 接口修改时主动 invalidate;
  多进程/多实例部署时靠 TTL(默认 300s)收敛,不保证秒级一致。
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict

import numpy as np

from . import db
from .embeddings import Embedder

logger = logging.getLogger(__name__)

# 语义路由命中阈值(bge-small-zh 实测:同域问句 ~0.55+,跨域一般 <0.4)
ROUTE_THRESHOLD = 0.45

# 内置路由名 → 图节点分支;其余意图一律走 kb_query 分支
# tool:工具调用路由(时间/计算器/网页抓取/知识库统计 + MCP 扩展),
# 可在设置页整体关闭——关闭后 _load 跳过该路由,语义层不再命中
BUILTIN_ROUTES = {"chitchat", "out_of_scope", "tool"}

# 路由实例缓存上限(超出按 LRU 逐出)与 TTL(秒)
CACHE_MAX_KBS = int(os.environ.get("ROUTER_CACHE_MAX_KBS", "64"))
CACHE_TTL = float(os.environ.get("ROUTER_CACHE_TTL", "300"))

# ---------------------------------------------------------------- 规则层
# 最简单的确定性场景不必走向量/LLM:短问候、明确的越界请求、敏感词。
# 规则只做"高置信短路",拿不准的一律放行给语义路由,避免误拦。

_GREETING_WORDS = {
    "你好", "您好", "hi", "hello", "嗨", "喂", "早上好", "上午好", "下午好",
    "晚上好", "早安", "晚安", "再见", "拜拜", "谢谢", "谢谢你", "感谢",
    "辛苦了", "在吗", "在么", "你是谁", "who are you",
}

# 明确越界的高频请求前缀/关键词(如写代码、翻译、闲聊时事)。
# 这是全局默认值;每个知识库可在 retrieval 配置的 oos_words 中覆盖
# (见 agent.py classify_intent)——"什么算超范围"取决于库的领域。
DEFAULT_OOS_WORDS = (
    "帮我写", "写一段代码", "写个代码", "翻译成", "翻译一下", "讲个笑话",
    "今天天气", "明天天气", "天气预报", "股票行情", "最新新闻", "算命",
)

# 敏感词(演示版):命中直接拒答。生产环境应外置为可维护的词表/审核服务。
SENSITIVE_WORDS = tuple(
    w for w in os.environ.get("SENSITIVE_WORDS", "").split(",") if w.strip()
)


def rule_route(query: str, oos_words: tuple | list | None = None) -> dict | None:
    """前置规则层:问候 / 明确越界 / 敏感词,命中即返回,source='rule'。

    oos_words=None 时用全局默认词表;传入列表(含空列表)时按知识库
    自定义词表判定——词表属于每库"检索设置",由 classify_intent 注入。
    敏感词表不开放配置(平台级安全底线),仍走全局 SENSITIVE_WORDS。
    """
    q = query.strip().lower().strip("?？!！.。~～ ")
    if not q:
        return None
    # 敏感词:任何位置命中都拒绝(最高优先级)
    for w in SENSITIVE_WORDS:
        if w.lower() in q:
            logger.info("rule_router 命中敏感词,直接拒答: %r", w)
            return {"name": "out_of_scope", "label": "敏感内容", "score": 1.0,
                    "boost_keywords": [], "source": "rule", "rule": "sensitive"}
    # 短消息精确问候(长度限制防止误伤"你好,请问休学手续..."这类复合句)
    if len(q) <= 8 and (q in _GREETING_WORDS or
                        any(q == w or q.startswith(w) and len(q) <= len(w) + 2
                            for w in _GREETING_WORDS)):
        return {"name": "chitchat", "label": "闲聊", "score": 1.0,
                "boost_keywords": [], "source": "rule", "rule": "greeting"}
    # 明确越界关键词(默认词表或每库自定义词表)
    for w in (DEFAULT_OOS_WORDS if oos_words is None else oos_words):
        if w in q:
            return {"name": "out_of_scope", "label": "超出范围", "score": 1.0,
                    "boost_keywords": [], "source": "rule", "rule": "blacklist"}
    return None


def extract_boost_keywords(text: str, top_k: int = 6) -> list[str]:
    """轻量关键词提取(jieba TF-IDF),用于 LLM 兜底分支的检索增强。

    语义路由未命中时意图关键词为空,但难分类的问题往往也最容易检索跑偏;
    从(改写前的)原始问题里提取实义词注入 BM25,作为兜底的关键词加权。
    """
    try:
        import jieba.analyse
        tags = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
        return [t for t in tags if len(t) >= 2]
    except Exception as e:  # jieba 不可用时降级为空,不阻断主链路
        logger.warning("关键词提取失败,跳过 BM25 加权: %s", e)
        return []


class SemanticRouter:
    """进程内语义路由器:utterance 向量缓存(LRU + TTL) + 余弦匹配。

    缓存键为 kb_id,值为 (载入时间, [(intent_dict, 归一化向量矩阵)]);
    命中缓存时不再访问数据库——配置变更通过 invalidate() 主动失效,
    TTL 作为多实例场景的兜底一致性窗口。
    """

    _instance: "SemanticRouter | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache: OrderedDict[int, tuple[float, list[tuple[dict, np.ndarray]]]] = \
            OrderedDict()
        self._cache_lock = threading.Lock()

    @classmethod
    def get(cls) -> "SemanticRouter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = SemanticRouter()
            return cls._instance

    # ---------------- 配置加载(仅缓存未命中/过期时执行,会访问 DB 并编码)

    def _load(self, kb_id: int) -> list[tuple[dict, np.ndarray]]:
        intents = db.list_intents(kb_id, include_global=True)
        # 工具调用总开关(设置页):关闭时 tool 路由不参与语义匹配,
        # 相关提问自然落回 LLM 兜底分类(其提示词同样不含 tool 类)
        from . import config as app_config   # 延迟导入,避免模块级环
        tools_on = bool(app_config.load_settings().get("tools_enabled", True))
        emb = Embedder.get()
        out: list[tuple[dict, np.ndarray]] = []
        seen_names: set[str] = set()
        for it in intents:
            if it["name"] == "tool" and not tools_on:
                continue
            name = it["name"]
            if name in seen_names:
                # kb 级与全局(kb_id=0)允许同名时,kb 级在前(list_intents 排序),
                # 后者会被遮蔽——属于配置隐患,打日志提醒。
                logger.warning("意图 %r 重复定义(kb_id=%s),后者被遮蔽,请检查配置",
                               name, it.get("kb_id"))
                continue
            seen_names.add(name)
            utt = [u for u in it.get("utterances", []) if u.strip()]
            if not utt:
                logger.warning("意图 %r(kb_id=%s)没有有效示例句,语义路由永远不会命中,已跳过",
                               name, it.get("kb_id"))
                continue
            mat = np.array([emb.embed_plain(u) for u in utt], dtype=np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
            out.append((it, mat))
        return out

    def _routes_for(self, kb_id: int) -> list[tuple[dict, np.ndarray]]:
        now = time.monotonic()
        with self._cache_lock:
            hit = self._cache.get(kb_id)
            if hit and now - hit[0] < CACHE_TTL:
                self._cache.move_to_end(kb_id)           # LRU: 标记最近使用
                return hit[1]
        # 进程内未命中:cluster 档查共享缓存(Redis),命中则免 DB 免编码
        from . import cache
        shared = cache.kv_get(f"router:{kb_id}")
        if shared is not None:
            routes = [(it, np.asarray(mat, dtype=np.float32)) for it, mat in shared]
        else:
            # 缓存未命中:在锁外加载(编码耗时,不阻塞其他 kb 的路由)
            routes = self._load(kb_id)
            cache.kv_set(f"router:{kb_id}",
                         [(it, mat) for it, mat in routes], ttl=CACHE_TTL)
        with self._cache_lock:
            self._cache[kb_id] = (now, routes)
            self._cache.move_to_end(kb_id)
            while len(self._cache) > CACHE_MAX_KBS:      # LRU 逐出最久未用
                self._cache.popitem(last=False)
        return routes

    @classmethod
    def invalidate(cls, kb_id: int | None = None):
        """配置变更后主动失效(CRUD 接口调用);kb_id=None 清空全部。
        进程内 + 共享缓存双失效,保证多实例一致。"""
        r = cls.get()
        from . import cache
        with r._cache_lock:
            if kb_id is None:
                r._cache.clear()
                cache.kv_delete("router:0")   # 全局路由在 kb_id=0 的键下
            else:
                r._cache.pop(kb_id, None)
                cache.kv_delete(f"router:{kb_id}")

    # ---------------- 路由

    def route(self, kb_id: int, query: str,
              threshold: float = ROUTE_THRESHOLD) -> dict | None:
        """命中返回 {name,label,score,boost_keywords,source:'semantic'},否则 None。

        注意:这里只嵌入用户最后一句话(query),不携带对话历史——
        多轮指代场景请依赖上层的 LLM 兜底分类(见模块 docstring 的设计取舍)。
        """
        routes = self._routes_for(kb_id)
        if not routes:
            return None
        q = np.array(Embedder.get().embed_plain(query), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        best: dict | None = None
        for it, mat in routes:
            score = float((mat @ q).max())          # 取该意图示例句中的最高分
            if best is None or score > best["score"]:
                best = {"name": it["name"], "label": it["label"], "score": score,
                        "boost_keywords": it.get("boost_keywords", [])}
        if best and best["score"] >= threshold:
            best["score"] = round(best["score"], 4)
            best["source"] = "semantic"
            if best["name"] not in BUILTIN_ROUTES:
                # 自定义意图按设计映射到 kb_query 分支;打 info 日志便于
                # 后台排查"配置写了自定义标签但走了知识库问答"的疑惑。
                logger.info("语义路由命中自定义意图 %r(%.4f),映射到 kb_query 分支",
                            best["name"], best["score"])
            return best
        return None
