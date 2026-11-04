"""业务系统数据对接(对应规格书 1.5 工具管理:自定义工具/API 插件/数据库查询)。

两层设计:
- 配置层:管理端在「工具管理 → 数据对接」页维护连接器(REST API 或只读数据库),
  保存前可点「测试连接」实时预览返回数据(行/列截断,防止大结果集刷屏);
- 运行层:启用的连接器自动注册为智能体工具(工具名 biz_<name>),与内置工具、
  MCP 工具同路径进入意图识别与 ReAct 循环;每次调用携带调用者身份写审计日志。

安全口径:
- 数据库连接器只读:SQL 文本级校验(仅 SELECT/WITH 开头、禁写库关键字、
  单语句),强制 LIMIT 行数上限;连接串中的账号权限由管理员在源库侧收敛。
- REST 连接器仅允许 http/https;响应体限流读取(256 KB),超时上限 30s。
- 业务系统多在内网,内网地址属正常目标,不做网段拦截。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from . import db

logger = logging.getLogger(__name__)

# 行/列/单元格截断:预览与工具输出共用,防止超长结果挤爆模型上下文与前端
MAX_ROWS = 200          # 单次查询硬上限(配置可更小)
PREVIEW_ROWS = 20       # 「测试连接」预览行数
MAX_COLS = 12
CELL_MAX = 200          # 单元格文本截断
TOOL_RESULT_MAX = 3000  # 工具回写模型的文本上限
REST_BODY_MAX = 256 * 1024

# SQL 注入/写库防护:剔除注释与字符串字面量后做关键字扫描
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|replace|"
    r"attach|detach|pragma|vacuum|execute|exec|call|merge|into|handler|"
    r"lock|unlock|shutdown)\b", re.I)


class ConnectorError(Exception):
    """业务可读错误:直接展示给管理端预览,或作为工具异常进入调用日志。"""


# ---------------------------------------------------------------- 配置校验

_PARAM_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,31}$")


def validate_config(ctype: str, cfg: dict) -> dict:
    """规范化并校验连接器配置;非法配置抛 ConnectorError(由路由层转 400)。"""
    if not isinstance(cfg, dict):
        raise ConnectorError("配置必须是 JSON 对象")
    cfg = dict(cfg)
    params = cfg.get("params") or []
    if not isinstance(params, list) or len(params) > 16:
        raise ConnectorError("参数定义 params 需为数组,且不超过 16 个")
    norm_params = []
    seen = set()
    for p in params:
        if not isinstance(p, dict) or not _PARAM_NAME_RE.fullmatch(str(p.get("name", ""))):
            raise ConnectorError("参数定义需包含合法 name(字母/下划线开头)")
        if p["name"] in seen:
            raise ConnectorError(f"参数名重复:{p['name']}")
        seen.add(p["name"])
        ptype = p.get("type", "string")
        if ptype not in ("string", "number", "integer", "boolean"):
            ptype = "string"
        norm_params.append({
            "name": p["name"], "type": ptype,
            "label": str(p.get("label") or p["name"])[:32],
            "description": str(p.get("description") or "")[:200],
            "required": bool(p.get("required", True)),
            "default": p.get("default", ""),
        })
    cfg["params"] = norm_params

    if ctype == "rest":
        method = str(cfg.get("method", "GET")).upper()
        if method not in ("GET", "POST"):
            raise ConnectorError("REST 连接器仅支持 GET/POST")
        url = str(cfg.get("url") or "").strip()
        if not re.match(r"^https?://", url):
            raise ConnectorError("REST 连接器 URL 必须以 http:// 或 https:// 开头")
        if len(url) > 500:
            raise ConnectorError("URL 过长")
        headers = cfg.get("headers") or {}
        if not isinstance(headers, dict) or len(headers) > 20:
            raise ConnectorError("headers 需为对象,且不超过 20 项")
        cfg.update(method=method, url=url,
                   headers={str(k)[:64]: str(v)[:500] for k, v in headers.items()},
                   timeout=min(max(int(cfg.get("timeout") or 10), 1), 30))
    elif ctype == "database":
        db_url = str(cfg.get("db_url") or "").strip()
        if not re.match(r"^(sqlite|mysql\+\w+|postgresql(\+\w+)?)://", db_url):
            raise ConnectorError(
                "数据库连接串仅支持 sqlite:// / mysql+驱动:// / postgresql://")
        query = str(cfg.get("query") or "").strip()
        _check_readonly_sql(query)   # 校验失败抛 ConnectorError
        cfg.update(db_url=db_url, query=query,
                   max_rows=min(max(int(cfg.get("max_rows") or 50), 1), MAX_ROWS))
    else:
        raise ConnectorError("连接器类型仅支持 rest / database")
    return cfg


def _strip_sql(sql: str) -> str:
    """剔除注释与字符串字面量,供关键字扫描(避免误杀字面量里的敏感词)。"""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return sql


def _check_readonly_sql(query: str) -> None:
    if not query or len(query) > 2000:
        raise ConnectorError("SQL 查询为空或过长(>2000 字符)")
    stripped = _strip_sql(query).strip()
    if not re.match(r"^(select|with)\b", stripped, re.I):
        raise ConnectorError("数据库连接器仅允许只读查询(SELECT/WITH 开头)")
    if ";" in stripped.rstrip(";"):
        raise ConnectorError("仅允许单条 SQL 语句")
    m = _FORBIDDEN_SQL.search(stripped)
    if m:
        raise ConnectorError(f"SQL 含只读策略禁止的关键字:{m.group(0).upper()}")


# ---------------------------------------------------------------- 参数模板

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]{0,31})\}")


def _render_template(value: Any, args: dict) -> Any:
    """字符串中的 {param} 占位符替换;非字符串(数字/嵌套 JSON)递归处理。"""
    if isinstance(value, str):
        def _sub(m):
            key = m.group(1)
            if key not in args:
                raise ConnectorError(f"缺少参数:{key}")
            return str(args[key])
        return _PLACEHOLDER_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _render_template(v, args) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_template(v, args) for v in value]
    return value


def _fill_args(cfg: dict, args: dict) -> dict:
    """应用默认值 + 必填校验;未知参数直接忽略(防止模型幻觉参数进入请求)。"""
    out = {}
    for p in cfg.get("params") or []:
        v = (args or {}).get(p["name"], p.get("default", ""))
        if (v is None or v == "") and p.get("required", True):
            raise ConnectorError(f"缺少必填参数:{p['name']}({p.get('label', '')})")
        out[p["name"]] = v if v is not None else ""
    return out


# ---------------------------------------------------------------- 结果规整

def _cell(v: Any) -> str:
    if v is None:
        return ""
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s[:CELL_MAX]


def _normalize_rows(data: Any, max_rows: int) -> dict:
    """把任意 JSON 规整为 {columns, rows, truncated} 表格,预览/工具输出共用。"""
    rows_obj = None
    if isinstance(data, list):
        rows_obj = data
    elif isinstance(data, dict):
        # 常见包裹形态:{data:[...]}/{rows:[...]}/{list:[...]}/{result:[...]}
        for k in ("data", "rows", "list", "result", "records", "items"):
            if isinstance(data.get(k), list):
                rows_obj = data[k]
                break
        if rows_obj is None:   # 纯对象 → 键值两列
            items = list(data.items())[: max_rows + 1]
            return {"columns": ["字段", "值"],
                    "rows": [[str(k)[:64], _cell(v)] for k, v in items[:max_rows]],
                    "truncated": len(items) > max_rows}
    if rows_obj is None:
        return {"columns": ["response"],
                "rows": [[_cell(data)]], "truncated": False}
    dict_rows = [r for r in rows_obj if isinstance(r, dict)]
    if dict_rows:
        cols: list[str] = []
        for r in dict_rows[:50]:
            for k in r.keys():
                if k not in cols:
                    cols.append(str(k))
                if len(cols) >= MAX_COLS:
                    break
        return {"columns": cols,
                "rows": [[_cell(r.get(c)) for c in cols]
                         for r in dict_rows[:max_rows]],
                "truncated": len(dict_rows) > max_rows}
    return {"columns": ["value"],
            "rows": [[_cell(v)] for v in rows_obj[:max_rows]],
            "truncated": len(rows_obj) > max_rows}


def _to_markdown(table: dict) -> str:
    """工具回写模型的紧凑表格(Markdown),整体超长时尾部截断。"""
    cols, rows = table["columns"], table["rows"]
    if not rows:
        return "(查询成功,无数据行)"
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    if table.get("truncated"):
        lines.append(f"(仅显示前 {len(rows)} 行,更多请缩小查询范围)")
    out = "\n".join(lines)
    return out[:TOOL_RESULT_MAX] + "\n…(结果过长已截断)" if len(out) > TOOL_RESULT_MAX else out


# ---------------------------------------------------------------- 执行

def _run_rest(cfg: dict, args: dict) -> dict:
    url = _render_template(cfg["url"], args)
    headers = _render_template(cfg.get("headers") or {}, args)
    method = cfg.get("method", "GET")
    kwargs: dict[str, Any] = {"headers": headers,
                              "timeout": cfg.get("timeout", 10), "stream": True}
    if method == "POST":
        kwargs["json"] = _render_template(cfg.get("body") or {}, args)
    try:
        resp = requests.request(method, url, **kwargs)
    except requests.RequestException as e:
        raise ConnectorError(f"请求失败:{type(e).__name__} {e}") from e
    chunks, size = [], 0
    try:
        for chunk in resp.iter_content(8192):
            chunks.append(chunk)
            size += len(chunk)
            if size > REST_BODY_MAX:
                break
    finally:
        resp.close()
    body = b"".join(chunks)
    if resp.status_code >= 400:
        raise ConnectorError(f"接口返回 HTTP {resp.status_code}:{body[:200].decode('utf-8', 'ignore')}")
    try:
        data = json.loads(body.decode("utf-8", "ignore"))
    except ValueError:
        return {"columns": ["response"],
                "rows": [[body.decode("utf-8", "ignore")[:2000]]],
                "truncated": size > REST_BODY_MAX}
    table = _normalize_rows(data, PREVIEW_ROWS)
    table["truncated"] = table["truncated"] or size > REST_BODY_MAX
    return table


def _run_database(cfg: dict, args: dict) -> dict:
    query = cfg["query"].strip().rstrip(";")
    max_rows = min(int(cfg.get("max_rows") or 50), MAX_ROWS)
    if not re.search(r"\blimit\b", _strip_sql(query), re.I):
        query = f"{query} LIMIT {max_rows + 1}"   # 多取 1 行判定截断
    try:
        # connect_timeout 是 MySQL/PG 驱动参数,sqlite 驱动不认识,按方言区分
        kwargs: dict[str, Any] = {"poolclass": NullPool}
        if not cfg["db_url"].startswith("sqlite"):
            kwargs["connect_args"] = {"connect_timeout": 10}
        eng = create_engine(cfg["db_url"], **kwargs)
    except Exception as e:
        raise ConnectorError(f"数据库连接串无效:{e}") from e
    try:
        with eng.connect() as conn:
            result = conn.execute(text(query), args)
            cols = [str(c) for c in list(result.keys())[:MAX_COLS]]
            rows = [[_cell(v) for v in r[:MAX_COLS]]
                    for r in result.fetchmany(max_rows + 1)]
    except ConnectorError:
        raise
    except Exception as e:
        raise ConnectorError(f"查询执行失败:{type(e).__name__} {e}") from e
    finally:
        eng.dispose()
    return {"columns": cols, "rows": rows[:max_rows],
            "truncated": len(rows) > max_rows}


def run_connector(cfg: dict, args: dict) -> dict:
    """执行连接器并返回 {columns, rows, truncated};失败抛 ConnectorError。"""
    args = _fill_args(cfg, args)
    if cfg.get("type") == "database" or "db_url" in cfg:
        return _run_database(cfg, args)
    return _run_rest(cfg, args)


def preview_connector(ctype: str, cfg: dict, args: dict) -> dict:
    """「测试连接」入口:校验 + 执行 + 计时,结果/错误统一结构化返回。"""
    t0 = time.time()
    try:
        cfg = validate_config(ctype, cfg)
        cfg["type"] = ctype
        table = run_connector(cfg, args or {})
        return {"ok": True, "duration_ms": int((time.time() - t0) * 1000), **table}
    except ConnectorError as e:
        return {"ok": False, "duration_ms": int((time.time() - t0) * 1000),
                "error": str(e)}
    except Exception as e:
        logger.exception("连接器预览异常")
        return {"ok": False, "duration_ms": int((time.time() - t0) * 1000),
                "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- 工具工厂

def connector_tools(user_id: int) -> list:
    """启用的连接器 → LangChain 工具表(每次请求现构造,配置改动即时生效)。

    与内置工具同口径:同步 invoke、30s 超时由 run_tool 统一控制;每次调用
    写审计日志(谁、何时、调了哪个业务接口、传了什么参数)。
    """
    from langchain_core.tools import StructuredTool
    from pydantic import create_model

    user = db.get_user(user_id) or {}
    tools = []
    for conn in db.list_connectors(enabled_only=True):
        cfg = conn.get("config") or {}
        fields = {}
        for p in cfg.get("params") or []:
            pytype = {"integer": int, "number": float,
                      "boolean": bool}.get(p.get("type"), str)
            desc = f"{p.get('label', '')}:{p.get('description', '')}".strip(":")
            if p.get("required", True) and p.get("default", "") == "":
                fields[p["name"]] = (pytype, ...)
            else:
                from pydantic import Field
                fields[p["name"]] = (
                    pytype, Field(default=p.get("default") or None, description=desc))
        try:
            schema = create_model(f"Biz_{conn['name']}_Args", **fields)
        except Exception as e:
            logger.error("连接器 %r 参数模型构建失败,已跳过: %s", conn["name"], e)
            continue

        def _make(c=conn, conf=cfg, sch=schema, uid=user_id, uname=user.get("username", "")):
            def _impl(**kwargs) -> str:
                db.audit_log(uid, uname, "connector_call",
                             f"{c['name']}({json.dumps(kwargs, ensure_ascii=False)[:300]})")
                conf2 = dict(conf, type=c["type"])
                table = run_connector(conf2, kwargs)
                return _to_markdown(table)
            return StructuredTool(
                name=f"biz_{c['name']}",
                description=(f"查询业务系统「{c.get('label') or c['name']}」的数据。"
                             f"{c.get('description') or ''}"
                             f"(来自业务系统数据对接,调用已审计)"),
                args_schema=sch,
                func=_impl,
            )
        tools.append(_make())
    return tools
