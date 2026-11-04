"""LangGraph RAG 问答代理(多知识库 + 意图识别 + 混合检索)。

流程:
  START → classify_intent(三级漏斗:规则层 → 语义路由 → LLM 兜底)
            ├─ kb_query    → rewrite_query(多轮指代改写) → retrieve(混合检索) → generate
            ├─ chitchat    → generate_chitchat(闲聊,简短引导回知识库)
            └─ out_of_scope→ refuse(礼貌拒答,声明知识库边界)

- 意图识别(对应规格书 1.4):
  1) 规则层 rule_router:问候词/明确越界关键词/敏感词,高置信短路,零开销;
  2) 语义路由:示例句向量余弦匹配(semantic-router 模式),0 次 LLM 调用;
     只看用户最后一句话、不看历史——多轮指代大概率低置信,刻意交给第 3 级;
  3) LLM 兜底:能看到与生成侧一致的 trim 后完整上下文,JSON 结构化输出,
     解析失败有正则兜底;同时用 jieba 提取问题实义词注入 BM25 检索加权。
- 查询改写:多轮对话中的指代/省略问题(如"那处分呢")先由 LLM 改写为独立检索问题
  (FastGPT 问题优化的简化版),首轮或关闭时直接透传。
- 混合检索:向量(Milvus) + BM25 全文 双路召回,RRF 融合,可选 BGE 精排,
  检索模式/阈值/权重按知识库配置(参考 RAGFlow/Dify/Langchain-Chatchat)。
- 生成:编号引用 prompt,回答内联标注 [1][2] 溯源编号,资料不足时拒答,
  最佳命中余弦过低时附低相关告警(kotaemon 风格)。
- 短时记忆:SqliteSaver checkpointer,thread_id = u{用户}_kb{知识库}_{会话} 三级隔离;
  生成前用 trim_messages 裁剪为最近 N 轮(默认 50 轮 = 100 条消息)。
- 父子块:检索命中子块时按 parent_id 去重,取父块全文组装上下文。
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import config as app_config, db
from .embeddings import Embedder
from .retrieval import Reranker, hybrid_search
from .router import (
    BUILTIN_ROUTES, ROUTE_THRESHOLD, SemanticRouter, extract_boost_keywords, rule_route,
)
from .store import VectorStore

logger = logging.getLogger(__name__)


def _patch_reasoning_content() -> None:
    """让 ChatOpenAI 保留推理模型的 reasoning_content(思考过程)。

    langchain-openai v1 只提取 OpenAI 标准字段,DeepSeek deepseek-reasoner、
    Qwen QwQ 等 OpenAI 兼容服务放在 delta/message 里的 reasoning_content
    会被丢弃(官方建议改用厂商子类)。为不引入额外依赖,这里对两个模块级
    转换函数做最小包装:仅当输入含 reasoning_content 时,转存到
    additional_kwargs —— 与 ChatDeepSeek 的做法一致,流式聚合时逐段拼接。
    任何一步失败都安静退回原行为:只是不显示思考过程,不影响正常回答。
    """
    try:
        from langchain_openai.chat_models import base as _lc_base
        orig_delta = _lc_base._convert_delta_to_message_chunk
        orig_msg = _lc_base._convert_dict_to_message

        def _delta(d, default_class, _orig=orig_delta):
            chunk = _orig(d, default_class)
            rc = d.get("reasoning_content")
            if rc and hasattr(chunk, "additional_kwargs"):
                chunk.additional_kwargs["reasoning_content"] = rc
            return chunk

        def _msg(d, _orig=orig_msg):
            msg = _orig(d)
            rc = d.get("reasoning_content")
            if rc and hasattr(msg, "additional_kwargs"):
                msg.additional_kwargs["reasoning_content"] = rc
            return msg

        _lc_base._convert_delta_to_message_chunk = _delta
        _lc_base._convert_dict_to_message = _msg
    except Exception as e:   # 依赖版本内部结构变化时静默失效
        logger.warning("reasoning_content 补丁未生效(不影响回答): %s", e)


_patch_reasoning_content()

SYSTEM_TEMPLATE = """你是“{kb_name}”智能问答助手,基于《{doc_name}》知识库回答问题。

【参考资料】(编号即引用编号)
{context}

回答要求:
1. 严格依据参考资料作答,不得编造或推测;资料不足以回答时,明确回复“根据现有资料无法回答该问题”,不要强行作答。
2. 引用资料中的事实时,在句末标注来源编号,格式如 [1]、[2];每条关键结论都要有出处。
3. 参考资料之间如有矛盾,指出矛盾并分别说明。
4. 用自己的语言简洁总结,不要整段照抄原文;涉及条款时注明条目编号(如“第X条”),表格内容以结构化方式呈现。
5. 对话历史中出现过的 [n] 编号仅属于历史回答,与本轮无关;请只依据本轮【参考资料】的编号引用。
6. 使用简体中文回答。"""

CHITCHAT_TEMPLATE = """你是“{kb_name}”智能问答助手,正在进行日常闲聊。
请简短友好地回应(1~3 句),并自然地引导用户提问与《{doc_name}》相关的制度、流程问题。
不要编造知识库内容。"""

INTENT_TEMPLATE = """你是意图分类器。请判断用户最后一条消息的意图类别。

知识库主题:《{doc_name}》({kb_name})。

类别定义:
- kb_query:需要从知识库检索规定、制度、流程、处分、条件等事实性内容的问题,或对上一轮知识库话题的追问
- chitchat:问候、寒暄、感谢、自我介绍等日常对话
- out_of_scope:与知识库主题无关、也无法用工具完成的请求(如写代码、翻译、新闻时事等)
- tool:需要调用工具才能完成的请求(查当前时间、日期推算、时间戳转换、数学计算、单位换算、个税估算、抓取网页内容、统计或列出知识库、查看文档目录等{tool_extra})

只输出 JSON,格式:{{"intent": "kb_query"}} 或 {{"intent": "chitchat"}} 或 {{"intent": "out_of_scope"}} 或 {{"intent": "tool"}}。不要输出任何其他内容。"""

INTENT_NAMES = ("kb_query", "chitchat", "out_of_scope", "tool")

REFUSAL_GEN_TEMPLATE = """你是“{kb_name}”智能问答助手的拒答模块。
用户的请求与《{doc_name}》无关,或触发了敏感词规则。请用 1~2 句简体中文礼貌拒绝,并自然地引导用户改问与该知识库相关的制度、流程问题。
不要编造知识库内容,不要回答用户的原始请求。"""

REWRITE_TEMPLATE = """你是查询改写器。根据对话历史,把用户最后一条消息改写为可独立检索的完整问题。

规则:
1. 补全指代(这/那/它/上面)与省略成分,保持原意,不臆造新事实。
2. 若原问题本身已完整独立,原样输出。
3. 只输出改写后的问题本身,不要任何解释或标点装饰。"""

INTENT_LABELS = {
    "kb_query": "知识库问答",
    "chitchat": "闲聊",
    "out_of_scope": "超出范围",
    "tool": "工具调用",
}

TOOL_SYSTEM = """你是“{kb_name}”智能问答助手的工具调用模块。
你可以使用提供的工具完成用户请求(查当前时间、日期推算、时间戳转换、数学计算、单位换算、个税估算、抓取网页、统计或列出知识库、查看文档目录等)。

要求:
1. 优先用工具获取事实结果,不要编造工具能给出的数据;工具结果不足时如实说明。
2. 回答基于工具返回内容,简洁准确,使用简体中文。
3. 不要主动向用户透露工具的参数细节与内部实现。{skills_section}"""

TOOL_MAX_ROUNDS = 4        # 工具调用轮数上限(技能流程:use_skill + 实际工具,预留一轮)
TOOL_RESULT_MAX = 4000     # 回传给模型的单条工具结果长度上限


def _enabled_skills() -> list[dict]:
    """读启用中的技能清单(30s 缓存,异常时安静降级为空——技能是增强项,
    它的故障绝不能拖垮意图分类与工具主链路)。"""
    try:
        from . import skills as skills_mod
        if not skills_mod.skills_on():
            return []
        return skills_mod.list_skills(enabled_only=True)
    except Exception:
        return []


def _tool_extra_hint() -> str:
    """LLM 兜底分类提示里的工具线索:技能/业务数据接口没有示例句向量可命中
    (语义路由的 tool 示例句只覆盖时间/计算/网页/统计),主要靠这行
    名称+描述把请求引向 tool 分支。"""
    parts = []
    items = _enabled_skills()
    if items:
        pairs = "、".join(f"{it['name']}({it['description'][:16]}…)"
                          if len(it["description"]) > 16
                          else f"{it['name']}({it['description']})"
                          for it in items[:6])
        parts.append(f"使用技能:{pairs}")
    try:   # 业务系统数据对接(规格书 1.5):启用的连接器按展示名提示
        from . import db as _db
        conns = _db.list_connectors(enabled_only=True)
        if conns:
            pairs = "、".join((c.get("label") or c["name"])[:12] for c in conns[:6])
            parts.append(f"查询业务系统:{pairs}")
    except Exception:
        pass
    return ("," + ";".join(parts)) if parts else ""


def _skills_section() -> str:
    """TOOL_SYSTEM 的技能段:渐进式披露——系统提示只放 name+description,
    完整指令由模型按需调 use_skill 拉取,不常驻上下文。"""
    items = _enabled_skills()
    if not items:
        return ""
    lines = "\n".join(f"- {it['name']}: {it['description']}" for it in items[:12])
    return ("\n\n可用技能(指令包):\n" + lines +
            "\n用户请求与某个技能描述匹配时,先调用 use_skill 加载其完整指令,"
            "再严格按指令执行;都不匹配时直接用自己的工具或知识回答,"
            "不要强行调用 use_skill。")


def _parse_intent_json(text: str) -> str:
    """解析 LLM 意图分类的 JSON 输出,容忍常见脏输出。

    链路:剥 markdown 围栏 → json.loads → 正则提取已知类别名 → 默认 kb_query。
    每一步失败都打日志,杜绝"静默吞错"(如模型输出 'kb_query。'、多词等)。
    """
    raw = text.strip()
    # 剥 ```json ... ``` 围栏
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    candidate = m.group(1).strip() if m else raw
    try:
        obj = json.loads(candidate)
        name = str(obj.get("intent", "")).strip().lower()
        if name in INTENT_NAMES:
            return name
        logger.warning("意图分类 JSON 中 intent 值非法: %r(原始输出 %r)", name, raw[:200])
    except (json.JSONDecodeError, AttributeError, TypeError):
        logger.warning("意图分类输出非 JSON,尝试结构化提取: %r", raw[:200])
    # 结构化提取:容忍 JSON 残缺(如单引号/尾逗号导致的 loads 失败),
    # 但不至于把解释性文本("这不是 out_of_scope,而是 kb_query")判反
    m = re.search(r"""["']intent["']\s*:\s*["']([a-z_]+)""", raw.lower())
    if m and m.group(1) in INTENT_NAMES:
        logger.info("意图分类经结构化提取识别为 %s", m.group(1))
        return m.group(1)
    # 最后兜底:找已知类别名(整体词匹配)。若出现多个,取最后出现的——
    # 解释性输出("这不是 out_of_scope,而是 kb_query")通常先否定后结论。
    found = [(m.end(), name) for name in INTENT_NAMES
             for m in re.finditer(rf"\b{name}\b", raw.lower())]
    if found:
        found.sort()
        name = found[-1][1]
        if len({n for _, n in found}) > 1:
            logger.warning("意图分类输出含多个类别名,取最后出现的 %s: %r",
                           name, raw[:200])
        else:
            logger.info("意图分类经关键词兜底识别为 %s", name)
        return name
    logger.warning("意图分类无法解析,默认 kb_query: %r", raw[:200])
    return "kb_query"

REFUSAL_TEMPLATE = (
    "抱歉,该问题超出了「{kb_name}」的服务范围。"
    "我只能解答与《{doc_name}》相关的制度、流程与规定问题,请换一个相关问题试试。"
)


class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    sources: list[dict]
    usage: dict
    intent: str           # 图分支名:kb_query / chitchat / out_of_scope
    intent_label: str     # 展示名:内置用中文标签,自定义路由用其 label(如"违纪处分")
    intent_score: float
    intent_source: str    # rule | semantic | llm
    route_name: str       # 语义路由原始命中名(如 discipline);非语义命中为空
    boost_keywords: list[str]
    context: str
    rewritten_query: str
    retrieval_debug: dict
    low_relevance: bool
    key_route: str        # 本次回答的 LLM 通道:primary | backup(埋点统计用)
    tool_calls: list[dict]  # 工具调用记录:[{name,label,args,status,preview}]


def thread_id_of(user_id: int, kb_id: int, session_id: str) -> str:
    return f"u{user_id}_kb{kb_id}_{session_id}"


class _Circuit:
    """主 Key 熔断器(进程内,参考 pybreaker 语义):连续 FAILURES 次失败后
    OPEN_SECS 秒内直连备用通道,避免每个请求先白等主 Key 超时。
    多实例下各实例独立熔断——主 Key 故障通常全局一致,近似等价且零依赖。"""

    FAILURES = 3
    OPEN_SECS = 60
    _fails = 0
    _opened_at = 0.0
    _lock = threading.Lock()

    @classmethod
    def is_open(cls) -> bool:
        with cls._lock:
            if cls._fails < cls.FAILURES:
                return False
            if time.monotonic() - cls._opened_at > cls.OPEN_SECS:
                cls._fails = 0          # 半开:放行一次探测,成功即闭合
                return False
            return True

    @classmethod
    def failure(cls) -> None:
        with cls._lock:
            cls._fails += 1
            cls._opened_at = time.monotonic()

    @classmethod
    def success(cls) -> None:
        with cls._lock:
            cls._fails = 0


class RAGAgent:
    _instance: "RAGAgent | None" = None
    _lock = threading.Lock()

    def __init__(self):
        self._graph = None
        self._checkpointer: SqliteSaver | None = None

    @classmethod
    def get(cls) -> "RAGAgent":
        with cls._lock:
            if cls._instance is None:
                cls._instance = RAGAgent()
            return cls._instance

    _retired: "RAGAgent | None" = None     # 上一代实例:挂起待关

    @classmethod
    def reload(cls) -> None:
        """丢弃单例,下次 get() 时重建。

        设置页改动(top_k/memory_turns/拒答开关等)是在 _build_graph 闭包里
        捕获的,不重建图就一直用旧值——设置页形同虚设。

        旧实例的 checkpointer 连接(sqlite conn / PG pool)不能立刻关:在途
        请求仍持有旧图,生成完写记忆那一步会崩。所以旧实例先"退休"挂起,
        下一次 reload 才关闭——泄漏有界(最多一代),在途请求有一整个两代
        间隔可以跑完。"""
        with cls._lock:
            prev_retired = cls._retired
            cls._retired = cls._instance
            cls._instance = None
        cls._close_checkpointer(prev_retired)

    @classmethod
    def _close_checkpointer(cls, inst: "RAGAgent | None") -> None:
        if inst is None:
            return
        try:
            conn = getattr(getattr(inst, "_checkpointer", None), "conn", None)
            # SqliteSaver.conn=sqlite3.Connection;PostgresSaver.conn=ConnectionPool,
            # 两者都有 close();pool 另有 closed 属性可判重
            if conn is not None and hasattr(conn, "close") \
                    and not getattr(conn, "closed", False):
                conn.close()
        except Exception as e:
            logger.warning("旧 checkpointer 关闭失败(不影响新图): %s", e)

    # ------------------------------------------------------------ 组件

    def _invoke(self, messages, streaming: bool = True,
                max_tokens: int | None = None, temperature: float = 0.2,
                model: str | None = None, tools: list | None = None):
        """带备用 Key 故障切换的 LLM 调用(路线二 七.5)。

        设计要点(参考 pybreaker / Dify 的模型降级模式):
        - 快速失败:配了备用 Key 时主通道 timeout=20s + max_retries=0,把时延预算
          留给备用通道——否则主 Key 故障时单次问答最坏 180s,先于网关超时死掉;
        - 熔断:主通道连续 3 次失败后 60s 内跳过主通道直连备用,避免每个请求
          都先白等一次主 Key 超时;
        - 双通道均失败按原异常抛出,由上层分支隔离兜底;
        - 通道标记写入 resp.response_metadata["_key_route"],供埋点区分主/备;
        - model/tools:工具节点用——model 覆盖设置页主模型(工具链路固定
          非思考模型),tools 非空时 bind_tools 后再发起调用。
        """
        s = app_config.load_settings()
        # getattr 兜底:settings 缺键时安静退化为"无备用"(安全默认值);
        # 而 is_cluster/CHECKPOINT_DB_URI 刻意保持直接访问——那两项缺失意味着
        # 部署档位错乱,必须立刻炸出来,不能静默降级成 standalone
        backup_key = s.get("deepseek_backup_api_key") or \
            getattr(app_config, "DEEPSEEK_BACKUP_API_KEY", "")
        backup_on = bool(backup_key)

        def _mk(api_key: str, timeout: int, retries: int):
            kw = {"max_tokens": max_tokens} if max_tokens else {}
            llm = ChatOpenAI(
                model=model or s["deepseek_model"], base_url=s["deepseek_base_url"],
                api_key=api_key, temperature=temperature, streaming=streaming,
                stream_usage=True, timeout=timeout, max_retries=retries, **kw)
            return llm.bind_tools(tools) if tools else llm

        primary_key = s["deepseek_api_key"]
        if not primary_key and not backup_on:
            raise RuntimeError("未配置 DeepSeek API Key")
        if not primary_key:
            # 只配了备用 Key:直连备用通道(主通道未启用,不计熔断)
            resp = _mk(backup_key, 60, 0).invoke(messages)
            resp.response_metadata["_key_route"] = "backup"
            return resp

        if backup_on and _Circuit.is_open():
            resp = _mk(backup_key, 60, 0).invoke(messages)
            resp.response_metadata["_key_route"] = "backup"
            return resp
        try:
            # 无备用 Key:保持原慢速重试策略(60s×2);有备用:快速失败
            resp = _mk(primary_key,
                       20 if backup_on else 60, 0 if backup_on else 1).invoke(messages)
            _Circuit.success()
            resp.response_metadata["_key_route"] = "primary"
            return resp
        except Exception as e:
            _Circuit.failure()
            if not backup_on:
                raise
            logger.warning("DeepSeek 主 Key 调用失败,切换备用 Key 重试: %s", e)
            logger.debug("主 Key 调用异常详情", exc_info=True)   # 折中:留 traceback 不刷屏
            resp = _mk(backup_key, 60, 0).invoke(messages)
            resp.response_metadata["_key_route"] = "backup"
            return resp

    @staticmethod
    def _kb_info(configurable: dict) -> tuple[int, str, str]:
        kb_id = int(configurable.get("kb_id", 1))
        # 短 TTL 缓存:同一请求的多个图节点共享,避免每请求 4 次 DB
        kb = db.get_kb_cached(kb_id) or {}
        doc_name = kb.get("doc_name") or kb.get("name") or "知识库"
        return kb_id, kb.get("name", "知识库"), doc_name

    @staticmethod
    def _content_text(content) -> str:
        """提取消息文本:纯文本直接返回;结构化 content(多模态 list)取 text 段。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content
                     if isinstance(p, dict) and p.get("type") in (None, "text")]
            return "\n".join(t for t in parts if t)
        return str(content or "")

    def _last_query(self, messages: list[BaseMessage]) -> str:
        for m in reversed(messages):
            if m.type == "human":
                return self._content_text(m.content)
        return ""

    def _build_graph(self):
        store = VectorStore.get()
        embedder = Embedder.get()
        settings = app_config.load_settings()
        top_k = int(settings["retrieve_top_k"])
        # 注意:trim_messages 的 token_counter=len 按消息条数计,"max_tokens"
        # 实际是"最多保留多少条消息";max(2,...) 兜底:配置为 0 时也必须
        # 保住当前这条用户消息,否则 LLM 只会收到一条 SystemMessage。
        max_messages = max(2, int(settings["memory_turns"]) * 2)

        # ------------------------------------------------ 上下文裁剪
        # 意图分类/生成/闲聊/拒答共用同一裁剪逻辑,保证各节点看到一致的历史
        def _trim(state: RAGState) -> list[BaseMessage]:
            return trim_messages(
                state["messages"],
                max_tokens=max_messages,     # 按消息条数裁剪(token_counter=len)
                strategy="last",
                token_counter=len,
                include_system=False,
                allow_partial=False,
                start_on="human",
            )

        def _usage_of(resp) -> dict:   # 普通嵌套函数;勿加 @staticmethod(嵌套作用域无类体)
            usage: dict = {}
            try:
                um = getattr(resp, "usage_metadata", None) or {}
                usage = {
                    "prompt_tokens": int(um.get("input_tokens", 0)),
                    "completion_tokens": int(um.get("output_tokens", 0)),
                    "total_tokens": int(um.get("total_tokens", 0)),
                }
                if not usage["total_tokens"]:
                    usage = resp.response_metadata.get("token_usage", {}) or {}
            except Exception:
                pass
            return usage

        # ------------------------------------------------ 意图识别(三级漏斗)
        def classify_intent(state: RAGState, config) -> dict:
            kb_id, kb_name, doc_name = self._kb_info(config["configurable"])
            kb = db.get_kb_cached(kb_id) or {}
            rc = kb.get("retrieval") or {}
            query = self._last_query(state["messages"])
            # 空查询(图内防御;api 层已拦):降级 chitchat 回引导语,代价最小——
            # 若走 kb_query 会空跑整条 改写→检索→生成 链并污染埋点(P1-5)
            if not query.strip():
                return {"intent": "chitchat", "intent_label": INTENT_LABELS["chitchat"],
                        "intent_score": 0.0, "intent_source": "empty",
                        "route_name": "", "boost_keywords": []}
            # 1) 规则层:问候/明确越界/敏感词,确定性短路(零向量零 LLM)
            #    超范围词表按库可配(检索设置 oos_words);未配置用全局默认
            rule = rule_route(query, oos_words=rc.get("oos_words"))
            if rule:
                intent = rule["name"]      # 规则层只产出内置名,直接作分支
                return {"intent": intent, "intent_label": rule["label"],
                        "intent_score": rule["score"], "intent_source": "rule",
                        "route_name": rule["name"], "boost_keywords": []}
            # 2) 语义路由:示例句向量匹配,0 次 LLM 调用(semantic-router 模式)
            #    命中阈值按库可配(检索设置 route_threshold,默认 ROUTE_THRESHOLD)——
            #    相似度绝对值依赖该库示例句与领域用词,全局一个值调不平所有库
            threshold = float(rc.get("route_threshold") or ROUTE_THRESHOLD)
            try:
                hit = SemanticRouter.get().route(kb_id, query, threshold=threshold)
            except Exception as e:
                logger.error("语义路由异常,降级到 LLM 分类: %s", e)
                hit = None
            # 缓存防护:路由缓存 TTL 内 settings 开关可能已变化——
            # 工具总开关关闭时,丢弃缓存中的 tool 命中,落到下方 LLM 兜底门控
            if hit and hit["name"] == "tool" and not \
                    app_config.load_settings().get("tools_enabled", True):
                hit = None
            if hit:
                # intent = 图分支名(自定义意图归一到 kb_query);intent_label =
                # 展示名(自定义意图保留自己的 label,如"违纪处分");route_name =
                # 原始命中名,用于埋点统计的意图分布——三者各司其职,不自相矛盾。
                name = hit["name"]
                intent = name if name in BUILTIN_ROUTES else "kb_query"
                return {"intent": intent, "intent_label": hit["label"],
                        "intent_score": hit["score"], "intent_source": "semantic",
                        "route_name": name,
                        "boost_keywords": hit.get("boost_keywords", [])}
            # 3) LLM 兜底分类(JSON 结构化输出;语义路由置信度不足时)
            #    上下文窗口与生成侧统一为 trim_messages:意图判定、改写、生成
            #    看到相同的历史,多轮指代场景不会出现认知割裂。
            recent = _trim(state)
            try:
                # 走 _invoke:主 Key 故障时备用 Key 同样覆盖意图分类(P0-3)
                resp = self._invoke(
                    [SystemMessage(content=INTENT_TEMPLATE.format(
                        kb_name=kb_name, doc_name=doc_name,
                        tool_extra=_tool_extra_hint()))] + recent,
                    streaming=False, max_tokens=32, temperature=0)
                intent = _parse_intent_json(self._content_text(resp.content))
            except Exception as e:
                logger.error("LLM 意图分类失败(双通道均不可用),按知识库问答处理: %s", e)
                intent = "kb_query"      # 分类失败时按知识库问答处理,保证可用性
            # 工具总开关关闭时,LLM 若仍判出 tool(提示词惯性),归一到拒答——
            # 此时平台确实无法承接这类请求,拒答话术如实声明边界
            if intent == "tool" and not app_config.load_settings().get("tools_enabled", True):
                intent = "out_of_scope"
            # 兜底分支没有意图关键词:从原始问题提取实义词注入 BM25 加权,
            # 难分类的问题往往也最容易检索跑偏,这里补一层(jieba TF-IDF,本地零成本)
            return {"intent": intent, "intent_label": INTENT_LABELS.get(intent, intent),
                    "intent_score": 0.0, "intent_source": "llm", "route_name": "",
                    "boost_keywords": extract_boost_keywords(query)}

        def route_intent(state: RAGState) -> str:
            return {"kb_query": "rewrite_query",
                    "chitchat": "generate_chitchat",
                    "out_of_scope": "refuse",
                    "tool": "tool_agent"}.get(state.get("intent", ""), "rewrite_query")

        # ------------------------------------------------ 查询改写(多轮指代消解)
        def rewrite_query(state: RAGState, config) -> dict:
            kb_id, _, _ = self._kb_info(config["configurable"])
            kb = db.get_kb_cached(kb_id) or {}
            rc = kb.get("retrieval") or {}
            query = self._last_query(state["messages"])
            history = [m for m in state["messages"][:-1] if m.type in ("human", "ai")]
            # 首轮或配置关闭时透传;改写失败静默回退原问题(FastGPT 风格)
            if not history or not rc.get("query_rewrite", True):
                return {"rewritten_query": query}
            try:
                # 走 _invoke:备用 Key 同样覆盖查询改写(P0-3)
                resp = self._invoke([SystemMessage(content=REWRITE_TEMPLATE)]
                                    + history[-6:]
                                    # 待改写内容用 user 角色:它是待处理文本而非系统指令,
                                    # 放 system 角色会让模型误当指令,改写质量不稳定
                                    + [HumanMessage(content=f"待改写的问题:{query}")],
                                    streaming=False, max_tokens=128, temperature=0)
                rewritten = self._content_text(resp.content).strip().strip("。\"' ")
                if rewritten and len(rewritten) <= 200:
                    return {"rewritten_query": rewritten}
            except Exception as e:
                logger.warning("查询改写失败,回退原问题: %s", e)
            return {"rewritten_query": query}

        # ------------------------------------------------ 混合检索
        def retrieve(state: RAGState, config) -> dict:
            kb_id, _, _ = self._kb_info(config["configurable"])
            kb = db.get_kb_cached(kb_id) or {}
            rc = {**{"mode": "semantic", "top_k": top_k}, **(kb.get("retrieval") or {})}
            query = state.get("rewritten_query") or self._last_query(state["messages"])
            try:
                vec = embedder.embed_query(query) if rc.get("mode") != "fulltext" else None
                hits, debug = hybrid_search(kb_id, query, vec, rc, store,
                                            boost_keywords=state.get("boost_keywords"))
                hits = hits or []               # 防御:检索契约被破坏时也不抛 TypeError
                if rc.get("rerank"):
                    # 精排是"锦上添花":单独捕获,失败保留已召回结果,不拖垮主链路
                    reranker = Reranker.get()
                    try:
                        hits = reranker.rerank(query, hits, int(rc.get("top_k", top_k)))
                        debug["reranked"] = reranker.available()
                        reranker.last_error = None   # 成功主动清,防上次残留(P1-6)
                    except Exception as e:
                        debug["reranked"] = False
                        debug["rerank_error"] = str(e)
                        logger.error("精排调用失败,保留 RRF 顺序: %s", e)
                    if reranker.last_error:      # 精排故障上报(告警可见)
                        debug["rerank_error"] = reranker.last_error
            except Exception as e:
                logger.error("检索节点整体异常 kb=%s: %s", kb_id, e)
                hits, debug = [], {"mode": rc.get("mode"), "error": str(e)}

            # 父子块:同一父块只取一次,上下文用父块全文;编号与 sources 顺序对齐
            seen_parents: set[int] = set()
            context_parts = []
            kept_sources = []
            for h in hits:
                meta = h.get("meta", {})
                pid = meta.get("parent_id")
                if pid is not None:
                    if pid in seen_parents:
                        continue
                    seen_parents.add(pid)
                    body = meta.get("parent_text") or h["text"]
                else:
                    body = h["text"]
                ctx_idx = len(kept_sources) + 1
                tag = "、".join(x for x in [meta.get("chapter", ""), meta.get("article", "")] if x)
                page = meta.get("page_start", "?")
                context_parts.append(f"[{ctx_idx}] {tag or '正文'}(第{page}页)\n{body}")
                kept_sources.append(h)
            return {
                "sources": kept_sources,
                "context": "\n\n".join(context_parts) or "(未检索到相关内容)",
                "retrieval_debug": debug,
                "low_relevance": bool(debug.get("low_relevance")),
            }

        # ------------------------------------------------ 生成
        # (_trim / _usage_of 已在文件上部定义,供分类/改写/检索/生成共用)
        def generate(state: RAGState, config) -> dict:
            _, kb_name, doc_name = self._kb_info(config["configurable"])
            sys_msg = SystemMessage(content=SYSTEM_TEMPLATE.format(
                kb_name=kb_name, doc_name=doc_name,
                context=state.get("context", "")))
            resp = self._invoke([sys_msg] + _trim(state), streaming=True, temperature=0.1)
            return {"messages": [resp], "usage": _usage_of(resp),
                    "key_route": resp.response_metadata.get("_key_route", "primary")}

        def generate_chitchat(state: RAGState, config) -> dict:
            _, kb_name, doc_name = self._kb_info(config["configurable"])
            sys_msg = SystemMessage(content=CHITCHAT_TEMPLATE.format(
                kb_name=kb_name, doc_name=doc_name))
            resp = self._invoke([sys_msg] + _trim(state), streaming=True)
            return {"messages": [resp], "usage": _usage_of(resp),
                    "key_route": resp.response_metadata.get("_key_route", "primary")}

        def refuse(state: RAGState, config) -> dict:
            _, kb_name, doc_name = self._kb_info(config["configurable"])
            # 默认走确定性模板:把用户的越界/敏感请求喂给 LLM,模型有概率
            # 直接作答——这正是拒答模块最不该发生的。LLM 拒答为可选增强
            # (REFUSAL_LLM=1 开启),且不回传用户原文与历史,只让模型写话术。
            if app_config.load_settings().get("refusal_llm"):
                try:
                    sys_msg = SystemMessage(content=REFUSAL_GEN_TEMPLATE.format(
                        kb_name=kb_name, doc_name=doc_name))
                    # streaming=False:api 层 SSE 只消费 generate/generate_chitchat
                    # 的 chunk,refuse 开流式只会多一个无用生成器(P1-4);
                    # max_tokens 上限防止模型写长篇"为什么不回答"
                    resp = self._invoke(
                        [sys_msg, HumanMessage(content="请生成一段礼貌的拒答话术。")],
                        streaming=False, temperature=0.3, max_tokens=128)
                    return {"messages": [resp], "usage": _usage_of(resp),
                            "key_route": resp.response_metadata.get("_key_route", "primary")}
                except Exception as e:
                    logger.warning("LLM 拒答生成失败,回退固定模板: %s", e)
            text = REFUSAL_TEMPLATE.format(kb_name=kb_name, doc_name=doc_name)
            return {"messages": [AIMessage(content=text)],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    # 模板分支不调 LLM,显式标记便于埋点区分(否则落库为空串)
                    "key_route": "template"}

        # ------------------------------------------------ 工具调用(一期)+ 技能(二期)
        # 不动原检索链:classify → tool_agent → END 独立分支。
        # 工具链路固定走非思考模型(设置页 tools_model,默认 deepseek-chat):
        # 思考模式 + 工具多轮回传 reasoning_content 的适配留作二期。
        # 中间过程消息(AIMessage+tool_calls / ToolMessage)只存在于本节点
        # 局部,不落 checkpoint——历史里只有"问题 + 最终回答",既干净又避免
        # 下轮 trim 截出孤儿 ToolMessage 触发 API 400。
        def tool_agent(state: RAGState, config) -> dict:
            from langchain_core.messages import ToolMessage
            try:
                from langgraph.config import get_stream_writer
                writer = get_stream_writer()
            except Exception:                      # 旧版 langgraph 无自定义流
                writer = lambda payload: None      # 降级:仅 done 事件携带记录

            kb_id, kb_name, _ = self._kb_info(config["configurable"])
            user_id = int(config["configurable"].get("user_id") or 0)
            s = app_config.load_settings()
            zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if not s.get("tools_enabled", True):
                return {"messages": [AIMessage(content=(
                            "工具调用功能当前未开启,请联系系统管理员在「设置」页启用。"))],
                        "usage": zero_usage, "key_route": "template", "tool_calls": []}

            from . import tools as tools_mod
            tools_list = tools_mod.get_tools(user_id, kb_id)
            tool_map = {t.name: t for t in tools_list}
            calls_log: list[dict] = []
            messages = [SystemMessage(content=TOOL_SYSTEM.format(
                kb_name=kb_name, skills_section=_skills_section()))] \
                + _trim(state)
            usage_total = dict(zero_usage)
            model = s.get("tools_model") or "deepseek-chat"

            def _accum_usage(resp) -> None:
                u = _usage_of(resp)
                for k in usage_total:
                    usage_total[k] += int(u.get(k, 0) or 0)

            resp = None
            for _round in range(TOOL_MAX_ROUNDS):
                # 工具选择轮全部非流式:tool_calls 必须完整拿到才能执行
                resp = self._invoke(messages, streaming=False, temperature=0,
                                    model=model, tools=tools_list)
                _accum_usage(resp)
                tcs = getattr(resp, "tool_calls", None) or []
                if not tcs:
                    break
                messages.append(resp)
                for tc in tcs:
                    name, args = tc.get("name", ""), tc.get("args") or {}
                    entry = {"name": name, "args": args, "status": "running",
                             "preview": ""}
                    calls_log.append(entry)
                    writer({"type": "tool_call", **entry})
                    t = tool_map.get(name)
                    try:
                        if t is None:
                            raise KeyError(f"未注册的工具: {name}")
                        result = tools_mod.run_tool(t, args)
                        entry["status"] = "ok"
                    except Exception as e:      # 工具失败不阻断:错误文本回喂模型
                        result = f"工具执行失败: {e}"
                        entry["status"] = "error"
                        logger.warning("工具 %s 执行失败: %s", name, e)
                    entry["preview"] = result[:200]
                    writer({"type": "tool_call", **entry})
                    messages.append(ToolMessage(
                        content=result[:TOOL_RESULT_MAX], tool_call_id=tc["id"]))
            else:
                # 达到轮数上限仍有未消化结果:兜底再调一次(不带工具)逼出文字回答
                logger.warning("工具调用达到 %d 轮上限,转入总结", TOOL_MAX_ROUNDS)

            if calls_log:
                # 有过工具调用:最终总结单独开流式,前端逐 token 出字
                resp = self._invoke(messages, streaming=True, temperature=0.2,
                                    model=model)
                _accum_usage(resp)
            return {"messages": [resp], "usage": usage_total,
                    "key_route": resp.response_metadata.get("_key_route", "primary"),
                    "tool_calls": calls_log}

        graph = StateGraph(RAGState)
        graph.add_node("classify_intent", classify_intent)
        graph.add_node("rewrite_query", rewrite_query)
        graph.add_node("retrieve", retrieve)
        graph.add_node("generate", generate)
        graph.add_node("generate_chitchat", generate_chitchat)
        graph.add_node("refuse", refuse)
        graph.add_node("tool_agent", tool_agent)
        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges("classify_intent", route_intent)
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        graph.add_edge("generate_chitchat", END)
        graph.add_edge("refuse", END)
        graph.add_edge("tool_agent", END)

        self._checkpointer = _make_checkpointer()
        # 显式建表(幂等):多实例并发启动时幂等 DDL 会争抢 migrations 锁,
        # 生产部署阶段一次性 setup 后可用 SKIP_CHECKPOINTER_SETUP=1 跳过(P1-8)
        if not getattr(app_config, "SKIP_CHECKPOINTER_SETUP", False):   # 缺省=执行 setup(安全默认)
            self._checkpointer.setup()
        self._graph = graph.compile(checkpointer=self._checkpointer)

    @property
    def graph(self):
        if self._graph is None:
            self._build_graph()
        return self._graph

    # ------------------------------------------------------------ 记忆查询

    def memory_turns(self, thread_id: str) -> int:
        try:
            state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
            msgs = state.values.get("messages", [])
            return sum(1 for m in msgs if m.type == "human")
        except Exception:
            return 0

    def history(self, thread_id: str) -> list[dict]:
        try:
            state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
            out = []
            for m in state.values.get("messages", []):
                if m.type in ("human", "ai"):
                    # _content_text:多模态结构化 content 取 text 段,str() 会变成 repr
                    out.append({"role": "user" if m.type == "human" else "assistant",
                                "content": self._content_text(m.content)})
            return out
        except Exception:
            return []

    def list_sessions(self, user_id: int, kb_id: int) -> list[dict]:
        """会话列表走独立的 chat_sessions 元数据表(按 user+kb 索引),
        不再遍历全量 checkpoints 表——SqliteSaver 的表为状态快照设计,
        会话量上来后全表 DISTINCT 扫描会成为瓶颈。"""
        try:
            return db.list_chat_sessions(user_id, kb_id)
        except Exception as e:
            logger.error("会话列表查询失败: %s", e)
            return []

    def delete_session(self, thread_id: str):
        _delete_thread_rows(thread_id)
        # thread_id = u{user}_kb{kb}_{session},同步删除会话元数据
        m = re.fullmatch(r"u(\d+)_kb(\d+)_(.+)", thread_id)
        if m:
            try:
                db.delete_chat_session(int(m.group(1)), int(m.group(2)), m.group(3))
            except Exception as e:
                logger.warning("会话元数据删除失败: %s", e)


_pg_pool = None
_pg_pool_lock = threading.Lock()


def _pg_pool_():
    """cluster 档 checkpoint 库连接池(路线二 P1-7):PostgresSaver 与
    _delete_thread_rows 共用,避免多 worker 下单连接瓶颈与频繁建连抖动。
    row_factory=dict_row 是 PostgresSaver 官方硬性要求;prepare_threshold=0
    规避 pgbouncer 类中间件下 prepared statement 冲突。"""
    global _pg_pool
    with _pg_pool_lock:
        if _pg_pool is None:
            import psycopg_pool
            from psycopg.rows import dict_row
            if not app_config.CHECKPOINT_DB_URI:
                raise RuntimeError("cluster 档必须配置 CHECKPOINT_DB_URI(如 rag_checkpoint 库)")
            _pg_pool = psycopg_pool.ConnectionPool(
                app_config.CHECKPOINT_DB_URI, min_size=1, max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": 0,
                        "row_factory": dict_row},
                open=True)
        return _pg_pool


def _make_checkpointer():
    """按部署档位创建 checkpointer(路线二 第一步):
    - standalone:SqliteSaver(零依赖,默认);
    - cluster:PostgresSaver + psycopg 连接池,接独立 checkpoint 库
      (CHECKPOINT_DB_URI,生产即 rag_checkpoint),多实例共享同一会话记忆。
    """
    if app_config.is_cluster():
        from langgraph.checkpoint.postgres import PostgresSaver
        return PostgresSaver(_pg_pool_())
    conn = sqlite3.connect(str(app_config.MEMORY_DB), check_same_thread=False,
                           timeout=30)   # 与删除会话等并发写错开,避免锁库
    return SqliteSaver(conn)


def _delete_thread_rows(thread_id: str) -> None:
    """删除某 thread 的全部 checkpoint 行(方言感知)。
    「表不存在」(旧版本差异)静默跳过;权限/连接类真实错误告警不吞(P2-13)。"""
    if app_config.is_cluster():
        import psycopg.errors
        with _pg_pool_().connection() as conn:
            for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
                try:
                    conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
                except psycopg.errors.UndefinedTable:
                    pass
                except Exception as e:
                    logger.warning("checkpoint 删除失败(%s): %s", table, e)
        return
    conn = sqlite3.connect(str(app_config.MEMORY_DB), check_same_thread=False,
                           timeout=30)   # 与 checkpointer 并发写错开,避免锁库
    try:
        # checkpoint_blobs 为新版 langgraph 的表,旧版不存在,逐表容错
        for table in ("checkpoints", "writes", "checkpoint_blobs"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError as e:
                if "no such table" not in str(e):
                    logger.warning("checkpoint 删除失败(%s): %s", table, e)
            except Exception as e:
                logger.warning("checkpoint 删除失败(%s): %s", table, e)
        conn.commit()
    finally:
        conn.close()
