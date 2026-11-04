"""工具调用:内置工具 + MCP 扩展入口(方案A:tool 意图 → 工具节点)。

设计要点:
- 内置工具(零 key、离线可用,共 10 个):当前时间 / 计算器 / 日期推算 /
  时间戳转换 / 单位换算 / 个税估算 / 网页抓取 / 知识库统计 / 知识库列表 /
  文档目录,均为进程内小函数,不依赖任何外部服务;
- MCP 扩展:设置页填 servers JSON,经 langchain-mcp-adapters 的
  MultiServerMCPClient 拉取远程/本地 server 的工具,与内置工具合并成
  统一注册表喂给 LangGraph 工具节点——对模型完全透明;
- 国内网络约束:MCP 协议本身不走墙,但选 server 时须注意其实际调用的
  API 国内可达(默认不预装任何需要 key 的 server);
- 故障隔离:MCP 加载失败静默降级为仅内置工具(60s 内不重试,防止
  每个请求都卡在坏 server 上);单个工具执行异常以 ToolException 文本
  回传给模型,不拖垮整条链路。
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import operator
import re
import time
from datetime import datetime, timedelta, timezone

from langchain_core.tools import ToolException, tool

from . import config as app_config

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_S = 30          # 单个工具执行上限
_MCP_OK_TTL = 300            # MCP 工具表缓存(配置不变时)
_MCP_ERR_TTL = 60            # 加载失败后的退避窗口


# ---------------------------------------------------------------- 内置工具

@tool("current_time")
def current_time() -> str:
    """获取当前日期与时间(北京时间)。用户问"现在几点/今天几号/星期几/
    什么日期"时使用,不需要参数。"""
    now = datetime.now(timezone(timedelta(hours=8)))
    week = "一二三四五六日"[now.weekday()]
    return now.strftime(f"%Y-%m-%d 星期{week} %H:%M:%S") + "(北京时间 UTC+8)"


# 计算器:AST 白名单求值,杜绝 eval 任意代码执行
_CALC_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_eval_node(node.operand))
    raise ToolException("表达式含有不支持的运算(仅支持 +-×÷ // % ** 与括号)")


@tool("calculator")
def calculator(expression: str) -> str:
    """数学计算。参数 expression 为算术表达式,支持 +-*/ // % ** 和括号,
    例如 "(128+36)*0.5"。任何数学/换算问题都应先用它算准,不要口算。"""
    expr = (expression or "").strip().replace("×", "*").replace("÷", "/")
    if not expr or len(expr) > 200:
        raise ToolException("表达式为空或过长")
    try:
        result = _eval_node(ast.parse(expr, mode="eval"))
    except ToolException:
        raise
    except Exception as e:
        raise ToolException(f"表达式无法计算: {e}") from e
    if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
        result = int(result)
    return f"{expr} = {result}"


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_WS_RE = re.compile(r"[ \t\r\f\v]+|\n{3,}")


@tool("web_fetch")
def web_fetch(url: str) -> str:
    """抓取网页正文(国内可访问的 http/https 页面)。参数 url 为完整网址。
    返回纯文本摘要(截断至 3000 字);需要总结/提取网页内容时使用。"""
    import urllib.request
    url = (url or "").strip()
    if not re.match(r"^https?://", url):
        raise ToolException("url 必须以 http:// 或 https:// 开头")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)",
        "Accept": "text/html,application/xhtml+xml,*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(1024 * 1024)                     # 限量读,防大页面打爆内存
            ctype = resp.headers.get("Content-Type", "")
    except Exception as e:
        raise ToolException(f"网页抓取失败: {e}") from e
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", ctype, re.I)
    if m:
        charset = m.group(1)
    try:
        html = raw.decode(charset, errors="replace")
    except LookupError:
        html = raw.decode("utf-8", errors="replace")
    html = _SCRIPT_RE.sub(" ", html)
    text = _TAG_RE.sub("\n", html)
    text = _WS_RE.sub(lambda m: "\n\n" if "\n" in m.group(0) else " ", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    body = "\n".join(lines)
    return body[:3000] + ("\n…(正文过长已截断)" if len(body) > 3000 else "")


# ---------------------------------------------------------------- 日期与时间

_DATE_FMTS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d")


def _parse_date(s: str) -> datetime:
    """宽松解析日期:支持 2025-03-01 / 2025/3/1 / 2025年3月1日,忽略时间部分。"""
    s = (s or "").strip()
    s = re.sub(r"[年月]", "-", s).replace("日", "").strip("- ")
    s = s.split()[0] if s.split() else s
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ToolException(f"无法识别日期: {s!r}(请用 YYYY-MM-DD 格式)")


def _week_cn(d: datetime) -> str:
    return "星期" + "一二三四五六日"[d.weekday()]


@tool("date_calc")
def date_calc(start: str, end: str = "", days: int = 0) -> str:
    """日期推算(不用于"现在几点",那是 current_time)。三种用法:
    1) 只传 start:返回该日期是星期几;
    2) 传 start 和 end:返回两个日期相差的天数;
    3) 传 start 和 days(整数,可为负):返回 start 之后(前)N 天的日期。
    参数 start/end 为日期,格式 YYYY-MM-DD(如 2025-03-01)。"""
    d0 = _parse_date(start)
    if (end or "").strip():
        d1 = _parse_date(end)
        diff = (d1 - d0).days
        return (f"{d0:%Y-%m-%d} 到 {d1:%Y-%m-%d} 相差 {abs(diff)} 天"
                f"({d1:%Y-%m-%d} 在{'后' if diff >= 0 else '前'})")
    if days:
        d1 = d0 + timedelta(days=int(days))
        return f"{d0:%Y-%m-%d} {'之后' if days > 0 else '之前'} {abs(int(days))} 天是 {d1:%Y-%m-%d} {_week_cn(d1)}"
    return f"{d0:%Y-%m-%d} 是 {_week_cn(d0)}"


@tool("timestamp_convert")
def timestamp_convert(value: str) -> str:
    """Unix 时间戳与北京时间互转。参数 value 为 10 位(秒)或 13 位(毫秒)
    数字时间戳,或为日期时间字符串(如 2025-03-01 12:30:00)。
    解读日志/数据库中的时间戳,或把日期转成时间戳时使用。"""
    v = (value or "").strip()
    tz = timezone(timedelta(hours=8))
    if re.fullmatch(r"\d{10}|\d{13}", v):
        ts = int(v)
        sec, ms = divmod(ts, 1000) if len(v) == 13 else (ts, 0)
        dt = datetime.fromtimestamp(sec, tz)
        suffix = f".{ms:03d}" if len(v) == 13 else ""
        return f"时间戳 {v} = {dt:%Y-%m-%d %H:%M:%S}{suffix} {_week_cn(dt)}(北京时间 UTC+8)"
    s = re.sub(r"[年月]", "-", v).replace("日", " ").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=tz)
            sec = int(dt.timestamp())
            return f"{dt:%Y-%m-%d %H:%M:%S}(北京时间)= 时间戳 {sec}(秒)/ {sec * 1000}(毫秒)"
        except ValueError:
            continue
    raise ToolException(f"无法识别: {v!r}(支持 10/13 位时间戳或 YYYY-MM-DD[ HH:MM:SS])")


# ---------------------------------------------------------------- 单位换算

# 类别 → {单位: 到基准单位的倍率};中文别名并入同一张表
_UNIT_TABLES = {
    "长度": ({"mm": 0.001, "毫米": 0.001, "cm": 0.01, "厘米": 0.01, "m": 1, "米": 1,
              "km": 1000, "公里": 1000, "千米": 1000, "inch": 0.0254, "英寸": 0.0254,
              "ft": 0.3048, "英尺": 0.3048, "mile": 1609.344, "英里": 1609.344,
              "nmi": 1852, "海里": 1852, "li": 500, "里": 500}, "m"),
    "重量": ({"mg": 1e-6, "毫克": 1e-6, "g": 1e-3, "克": 1e-3, "kg": 1, "千克": 1,
              "公斤": 1, "t": 1000, "吨": 1000, "斤": 0.5, "lb": 0.45359237,
              "磅": 0.45359237, "oz": 0.028349523125, "盎司": 0.028349523125}, "kg"),
    "面积": ({"m2": 1, "平方米": 1, "km2": 1e6, "平方公里": 1e6, "ha": 1e4,
              "公顷": 1e4, "mu": 2000 / 3, "亩": 2000 / 3, "sqft": 0.09290304,
              "平方英尺": 0.09290304}, "m2"),
    "体积": ({"ml": 1e-3, "毫升": 1e-3, "l": 1, "升": 1, "m3": 1000, "立方米": 1000,
              "gal": 3.785411784, "加仑": 3.785411784}, "L"),
    "存储": ({"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3,
              "tb": 1024 ** 4, "pb": 1024 ** 5}, "B"),
}
_TEMP_UNITS = {"c", "f", "k", "摄氏度", "华氏度", "开尔文"}


def _to_celsius(v: float, u: str) -> float:
    if u in ("c", "摄氏度"):
        return v
    if u in ("f", "华氏度"):
        return (v - 32) * 5 / 9
    return v - 273.15


def _from_celsius(v: float, u: str) -> float:
    if u in ("c", "摄氏度"):
        return v
    if u in ("f", "华氏度"):
        return v * 9 / 5 + 32
    return v + 273.15


@tool("unit_convert")
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算:长度/重量/面积/体积/温度/数据存储。参数 value 为数值,
    from_unit 与 to_unit 为单位(支持中文,如 公里/千克/亩/升/摄氏度,
    或英文缩写 km/kg/mu/L/C)。不含汇率换算(汇率请用 web_fetch 查实时价)。"""
    fu, tu = (from_unit or "").strip(), (to_unit or "").strip()
    flu, tlu = fu.lower(), tu.lower()
    if flu in _TEMP_UNITS and tlu in _TEMP_UNITS:
        c = _to_celsius(float(value), flu)
        return f"{value:g} {fu} = {_from_celsius(c, tlu):g} {tu}"
    for cat, (table, base) in _UNIT_TABLES.items():
        low = {k.lower(): v for k, v in table.items()}
        if flu in low and tlu in low:
            result = float(value) * low[flu] / low[tlu]
            return f"{value:g} {fu} = {result:g} {tu}({cat}换算)"
    raise ToolException(f"不支持或不同类别的单位: {fu} → {tu}")


# ---------------------------------------------------------------- 个税估算

# 居民个人工资薪金月度预扣率表(七级累进):(上限, 税率, 速算扣除数)
_TAX_BRACKETS = [
    (3000, 0.03, 0), (12000, 0.10, 210), (25000, 0.20, 1410),
    (35000, 0.25, 2660), (55000, 0.30, 4410), (80000, 0.35, 7160),
    (float("inf"), 0.45, 15160),
]


@tool("tax_calc")
def tax_calc(monthly_salary: float) -> str:
    """中国大陆工资薪金个税估算(月度预扣)。参数 monthly_salary 为税前月薪
    (元)。按 5000 元起征点与七级累进预扣率表计算,未计入社保公积金与
    专项附加扣除——精确税额以实际申报为准。估算个税或税后工资时使用;
    一般四则运算请用 calculator。"""
    salary = float(monthly_salary)
    if salary < 0 or salary > 1e8:
        raise ToolException("月薪数值不合理")
    taxable = salary - 5000
    if taxable <= 0:
        return (f"税前月薪 {salary:g} 元,未超过 5000 元起征点,预扣个税 0 元,"
                f"税后约 {salary:g} 元(未计社保公积金与专项附加扣除)")
    for cap, rate, deduct in _TAX_BRACKETS:
        if taxable <= cap:
            tax = taxable * rate - deduct
            break
    return (f"税前月薪 {salary:g} 元:应纳税所得额 {taxable:g} 元,适用税率 "
            f"{rate:.0%}、速算扣除数 {deduct:g},预扣个税 {tax:.2f} 元,"
            f"税后约 {salary - tax:.2f} 元(未计社保公积金与专项附加扣除)")


def _kb_stats_tool(user_id: int, kb_id: int):
    """知识库统计工具:闭包注入当前用户与知识库,保证统计口径不越权
    (只统计该用户可访问的知识库,私有库存在性不外泄)。"""
    from . import db
    from .store import VectorStore

    @tool("kb_stats")
    def _impl() -> str:
        """统计知识库平台数据:当前知识库的文档/索引块数量,以及你可访问的
        全部知识库列表与各自规模。用户问"有多少文档/知识库/块"时使用。"""
        user = db.get_user(user_id) or {}
        kbs = db.accessible_kbs(user) if user else []
        store = VectorStore.get()
        lines = []
        for kb in kbs:
            st = store.status(kb["id"])
            mark = " ← 当前知识库" if kb["id"] == kb_id else ""
            lines.append(
                f"- {kb['name']}({kb.get('doc_name') or '未上传文档'},"
                f"{'已建索引 ' + str(st.get('count', 0)) + ' 块' if st.get('indexed') else '未建索引'}){mark}")
        return (f"你可访问的知识库共 {len(kbs)} 个:\n" + "\n".join(lines)) if lines \
            else "你当前没有可访问的知识库"

    return _impl


def _kb_list_tool(user_id: int):
    """知识库列表工具:闭包注入当前用户,只列出其可访问的库(私有库存在性
    不外泄)。与 kb_stats 分工:本工具回答"有哪些/能用什么",kb_stats
    回答"有多少/多大"。"""
    from . import db

    @tool("kb_list")
    def _impl() -> str:
        """列出你可访问的知识库清单(名称、简介、公开/私有、负责人)。
        用户问"我有哪些知识库/能查哪些库/该用哪个库"时使用;
        要统计各库文档与索引块数量时改用 kb_stats。"""
        user = db.get_user(user_id) or {}
        kbs = db.accessible_kbs(user) if user else []
        if not kbs:
            return "你当前没有可访问的知识库"
        lines = []
        for kb in kbs:
            owner = db.get_user(kb["owner_id"])
            desc = (kb.get("description") or "").strip()
            lines.append(
                f"- {kb['name']}({'公开' if kb.get('visibility') == 'public' else '私有(授权)'}"
                f",负责人:{owner['display_name'] if owner else '未知'}"
                f",文档:{kb.get('doc_name') or '未上传'}"
                + (f",简介:{desc[:60]}" if desc else "") + ")")
        return f"你可访问的知识库共 {len(kbs)} 个:\n" + "\n".join(lines)

    return _impl


_CHAPTER_LINE_RE = re.compile(r"^(第[一二三四五六七八九十\d]{1,4}[章节编][ 　]?\S{0,30})\s*$")
_ARTICLE_ANY_RE = re.compile(r"第[一二三四五六七八九十百千零〇两\d]{1,7}条")


def _doc_outline_tool(kb_id: int):
    """文档目录工具:闭包注入当前知识库,从解析结果(full_text)扫描
    章/节标题与"第X条"分布,不依赖索引是否已构建。"""

    @tool("doc_outline")
    def _impl() -> str:
        """查看当前知识库文档的目录结构(第X章/第X节标题及各章包含的条目数)。
        用户问"这份文档有哪些章节/目录是什么/都讲了哪些方面"时使用;
        具体条文内容的问题不要用本工具(那是知识库检索的事)。"""
        from .parser import load_parsed
        doc = load_parsed(app_config.kb_dir(kb_id) / "document.json")
        if doc is None:
            raise ToolException("当前知识库尚未上传或解析文档")
        chapters: list[tuple[str, int]] = []   # (标题, 条数)
        loose_articles = 0
        for line in (doc.full_text or "").splitlines():
            s = line.strip()
            if not s:
                continue
            m = _CHAPTER_LINE_RE.match(s)
            if m:
                chapters.append((m.group(1), 0))
                continue
            hits = _ARTICLE_ANY_RE.findall(s)
            if hits:
                if chapters:
                    chapters[-1] = (chapters[-1][0], chapters[-1][1] + len(hits))
                else:
                    loose_articles += len(hits)
        total_articles = sum(n for _, n in chapters) + loose_articles
        if not chapters and not total_articles:
            return (f"《{doc.name}》共 {doc.page_count} 页,未识别到"
                    f"「第X章/第X条」式结构(可能是非法规类文档)")
        lines = [f"《{doc.name}》共 {doc.page_count} 页,目录结构:"]
        if loose_articles:
            lines.append(f"- (正文未分章部分)约 {loose_articles} 条")
        for title, n in chapters[:60]:
            lines.append(f"- {title}" + (f"(约 {n} 条)" if n else ""))
        if len(chapters) > 60:
            lines.append(f"- ……共 {len(chapters)} 个章节,其余略")
        return "\n".join(lines)[:2800]

    return _impl


@tool("use_skill")
def use_skill(name: str) -> str:
    """加载指定技能(Skill)的完整指令包。用户请求与系统提示中列出的某个
    技能描述匹配时,先调用本工具加载指令,然后严格按指令完成后续步骤。
    参数 name 为技能名(必须与清单中的名字完全一致)。"""
    from . import skills as skills_mod
    item = skills_mod.get_skill((name or "").strip())
    if item is None:
        raise ToolException(
            f"技能「{name}」不存在或未启用,请直接根据现有信息回答,或告知用户该技能不可用")
    # 指令正文显著标记,帮助模型区分"数据"与"指令";长度在加载时已受控
    return (f"【技能 {item['name']} 指令包开始】\n{item['body']}"
            f"\n【技能 {item['name']} 指令包结束,请立即按上述指令执行】")


def _skills_available() -> bool:
    """技能总开关开启且清单非空时,才把 use_skill 挂进注册表——
    没有技能可加载时挂着只会诱导空调用。"""
    from . import skills as skills_mod
    if not skills_mod.skills_on():
        return False
    try:
        return bool(skills_mod.list_skills(enabled_only=True))
    except Exception:
        return False


def builtin_tools(user_id: int, kb_id: int) -> list:
    """内置工具表(每次调用现构造:kb_stats/kb_list/doc_outline 闭包携带
    本次请求的上下文,保证权限口径不越权)。"""
    tools = [current_time, calculator, date_calc, timestamp_convert,
             unit_convert, tax_calc, web_fetch,
             _kb_stats_tool(user_id, kb_id), _kb_list_tool(user_id),
             _doc_outline_tool(kb_id)]
    if _skills_available():
        tools.append(use_skill)
    return tools


# ---------------------------------------------------------------- MCP 扩展

_mcp_cache: dict[str, tuple[float, list]] = {}   # config_hash → (到期时间, 工具表)


def _mcp_config() -> dict:
    """解析设置页的 MCP servers JSON,并做最小净化与 transport 推断。"""
    raw = str(app_config.load_settings().get("mcp_servers") or "").strip()
    if not raw:
        return {}
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("MCP servers 配置不是合法 JSON,已忽略: %s", e)
        return {}
    if not isinstance(cfg, dict):
        logger.error("MCP servers 配置应为 {名称: 连接} 的对象,已忽略")
        return {}
    out: dict = {}
    for name, conn in list(cfg.items())[:8]:               # 上限 8 个,防误配刷屏
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", str(name)) or not isinstance(conn, dict):
            logger.warning("MCP server 配置项非法,已跳过: %r", name)
            continue
        conn = dict(conn)
        if "url" in conn:
            # 未显式声明时按 URL 形态推断:/sse 结尾走 SSE,否则 streamable_http
            conn.setdefault("transport",
                            "sse" if str(conn["url"]).rstrip("/").endswith("/sse")
                            else "streamable_http")
        elif "command" in conn:
            conn.setdefault("transport", "stdio")
        else:
            logger.warning("MCP server %r 缺少 url/command,已跳过", name)
            continue
        out[str(name)] = conn
    return out


def _load_mcp_tools(cfg: dict) -> list:
    """同步上下文里取异步 MCP 工具表(本进程节点跑在线程池,无事件循环)。"""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient(cfg)
    return asyncio.run(asyncio.wait_for(client.get_tools(), timeout=30))


def mcp_tools() -> list:
    """MCP 工具表:成功缓存 300s;失败缓存空表 60s 退避,不阻断内置工具。"""
    cfg = _mcp_config()
    if not cfg:
        return []
    key = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    hit = _mcp_cache.get(key)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    try:
        tools = _load_mcp_tools(cfg)
    except ImportError:
        logger.error("langchain-mcp-adapters 未安装,MCP 工具不可用(内置工具不受影响)")
        tools = None
    except Exception as e:
        logger.error("MCP 工具加载失败(本轮仅用内置工具): %s", e)
        tools = None
    if tools is None:
        _mcp_cache[key] = (time.monotonic() + _MCP_ERR_TTL, [])
        return []
    for t in tools:   # MCP 工具统一补超时/异常说明,便于节点层日志定位
        t.description = (t.description or "") + f"(来自 MCP server)"
    _mcp_cache[key] = (time.monotonic() + _MCP_OK_TTL, tools)
    logger.info("MCP 工具已加载: %s", [t.name for t in tools])
    return tools


def get_tools(user_id: int, kb_id: int) -> list:
    """统一工具注册表 = 内置工具 + 业务数据对接工具 + MCP 工具(对模型透明)。

    业务系统连接器(规格书 1.5):启用中的连接器逐条注册为 biz_<name> 工具,
    与内置工具同路径参与意图识别与 ReAct 循环;单条连接器异常不拖垮整体。
    """
    tools = builtin_tools(user_id, kb_id)
    try:
        from . import connectors
        tools += connectors.connector_tools(user_id)
    except Exception as e:
        logger.error("业务数据对接工具装配失败(本轮不含连接器工具): %s", e)
    return tools + mcp_tools()


def run_tool(t, args: dict) -> str:
    """执行单个工具:MCP 工具仅实现异步接口,同步上下文里用事件循环跑;
    统一 30s 超时;结果一律转 str,异常向上抛由节点记入调用日志。"""
    if getattr(t, "coroutine", None) is not None:
        return str(asyncio.run(asyncio.wait_for(t.ainvoke(args), TOOL_TIMEOUT_S)))
    return str(t.invoke(args))
