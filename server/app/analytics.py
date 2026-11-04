"""埋点统计:问答事件、token 消耗、耗时(对应规格书 1.10)。

路线二改造:与业务数据共用 db.engine()——
- standalone:同库 SQLite(零依赖);
- cluster:IvorySQL,多实例写同一库,连接池由 db 层管理。
SQL 全部走 SQLAlchemy Core,无方言硬编码;timeseries 按日聚合在 Python 侧
分桶(数据量小),规避 SQLite strftime / PG date_trunc 的方言差异。
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from sqlalchemy import func, select

from . import db


def record(kind: str, session_id: str = "", question: str = "",
           answer_preview: str = "", usage: dict | None = None,
           latency_ms: int = 0, sources_n: int = 0, strategy: str = "",
           user_id: int = 0, username: str = "", kb_id: int = 0,
           kb_name: str = "", intent: str = "", key_route: str = ""):
    """写一条埋点;失败只告警(统计不应打断主流程)。"""
    usage = usage or {}
    try:
        with db.engine().begin() as conn:
            conn.execute(db.events.insert().values(
                ts=time.time(), kind=kind, session_id=session_id,
                question=question[:200], answer_preview=answer_preview[:200],
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                latency_ms=latency_ms, sources_n=sources_n, strategy=strategy,
                user_id=user_id, username=username, kb_id=kb_id,
                kb_name=kb_name, intent=intent, key_route=key_route))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("埋点写入失败: %s", e)


def summary() -> dict:
    e = db.events
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    chat = e.c.kind == "chat"
    with db.engine().connect() as conn:
        row = conn.execute(
            select(func.count(), func.coalesce(func.sum(e.c.total_tokens), 0),
                   func.coalesce(func.avg(e.c.latency_ms), 0)).where(chat)).first()
        today = conn.execute(
            select(func.count(), func.coalesce(func.sum(e.c.total_tokens), 0))
            .where(chat, e.c.ts >= today_start)).first()
        sessions = conn.execute(
            select(func.count(func.distinct(e.c.session_id))).where(chat)).scalar()
        builds = conn.execute(
            select(func.count()).where(e.c.kind == "build")).scalar()
    return {
        "total_questions": row[0], "total_tokens": row[1],
        "avg_latency_ms": round(row[2] or 0),
        "today_questions": today[0], "today_tokens": today[1],
        "sessions": sessions or 0, "builds": builds or 0,
    }


def timeseries(days: int = 14) -> list[dict]:
    """最近 N 天问答量/token 按日聚合(Python 分桶,方言中立)。"""
    start = (datetime.now() - timedelta(days=days - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    with db.engine().connect() as conn:
        rows = conn.execute(
            select(db.events.c.ts, db.events.c.total_tokens)
            .where(db.events.c.kind == "chat", db.events.c.ts >= start.timestamp())
        ).all()
    by_day: dict[int, list] = {}
    for ts, tokens in rows:
        day = int(datetime.fromtimestamp(ts).replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp())
        slot = by_day.setdefault(day, [0, 0])
        slot[0] += 1
        slot[1] += tokens or 0
    out = []
    for i in range(days):
        d = start + timedelta(days=i)
        q, t = by_day.get(int(d.timestamp()), (0, 0))
        out.append({"date": d.strftime("%m-%d"), "questions": q, "tokens": t})
    return out


def recent(limit: int = 15) -> list[dict]:
    e = db.events
    with db.engine().connect() as conn:
        rows = conn.execute(
            select(e.c.ts, e.c.session_id, e.c.question, e.c.answer_preview,
                   e.c.total_tokens, e.c.latency_ms, e.c.sources_n,
                   e.c.username, e.c.kb_name, e.c.intent)
            .where(e.c.kind == "chat").order_by(e.c.ts.desc()).limit(limit)).all()
    return [
        {
            "time": datetime.fromtimestamp(r[0]).strftime("%m-%d %H:%M"),
            "session_id": r[1][-8:], "question": r[2], "answer_preview": r[3],
            "tokens": r[4], "latency_ms": r[5], "sources_n": r[6],
            "username": r[7], "kb_name": r[8], "intent": r[9],
        }
        for r in rows
    ]


def by_intent() -> list[dict]:
    """意图分布统计(对应规格书 1.4 意图识别的埋点)。"""
    e = db.events
    with db.engine().connect() as conn:
        rows = conn.execute(
            select(e.c.intent, func.count())
            .where(e.c.kind == "chat", e.c.intent != "")
            .group_by(e.c.intent).order_by(func.count().desc())).all()
    return [{"intent": r[0], "count": r[1]} for r in rows]


def by_kb() -> list[dict]:
    """分知识库提问量统计。"""
    e = db.events
    with db.engine().connect() as conn:
        rows = conn.execute(
            select(e.c.kb_name, func.count(),
                   func.coalesce(func.sum(e.c.total_tokens), 0))
            .where(e.c.kind == "chat", e.c.kb_name != "")
            .group_by(e.c.kb_name).order_by(func.count().desc())).all()
    return [{"kb_name": r[0], "questions": r[1], "tokens": r[2]} for r in rows]
