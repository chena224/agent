#!/usr/bin/env python3
"""standalone → cluster 业务库迁移工具 (SQLite → IvorySQL/PostgreSQL)

用法(在 server/ 目录下):
    PYTHONPATH=. python3 scripts/migrate_sqlite_to_pg.py \
        --src sqlite:///data/users.db \
        --dst postgresql+psycopg://rag:rag@127.0.0.1:5432/rag_db

说明:
- 幂等:目标库中主键冲突的行跳过,可重复执行;
- 搬完自动修 SERIAL 自增序列(setval),避免显式 id 种子导致主键冲突;
- 只迁移业务库(users/knowledge_bases/grants/intents/chat_sessions/documents/
  audit_logs/events)。settings.json、Milvus 向量、LangGraph checkpoint 不在此列,
  处理建议见 README「部署模式」章节;
- 迁移期间建议停服,避免源库写入造成不一致。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # server/

from sqlalchemy import create_engine, func, select, text  # noqa: E402
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: E402

from app import db as dbm  # noqa: E402  复用业务层表定义,避免 schema 漂移

TABLES = [  # 按外键依赖顺序
    dbm.users,
    dbm.auth_sessions,
    dbm.knowledge_bases,
    dbm.kb_grants,
    dbm.kb_intents,
    dbm.chat_sessions,
    dbm.documents,
    dbm.audit_logs,
    dbm.events,
]


def _count(eng, table) -> int:
    with eng.connect() as conn:
        return conn.execute(select(func.count()).select_from(table)).scalar_one()


def migrate(src_url: str, dst_url: str) -> bool:
    if not dst_url.startswith(("postgresql://", "postgresql+psycopg://")):
        print("--dst 必须是 postgresql:// DSN (IvorySQL 兼容 PG 协议)", file=sys.stderr)
        return False
    src = create_engine(src_url)
    dst = create_engine(dst_url)

    dbm._meta.create_all(dst)   # 目标建表(幂等)
    dbm._ensure_columns(dst)    # 存量补列(与线上 init_db 同一套迁移)

    ok = True
    for table in TABLES:
        with src.connect() as conn:
            rows = [dict(r) for r in conn.execute(select(table)).mappings()]
        before = _count(dst, table)
        with dst.begin() as conn:
            for row in rows:
                conn.execute(
                    pg_insert(table).on_conflict_do_nothing().values(**row)
                )
        dst_total = _count(dst, table)
        inserted = dst_total - before               # 以行数差为准,不依赖 rowcount
        skipped = len(rows) - inserted
        status = "OK" if dst_total >= len(rows) else "MISMATCH"
        if status != "OK":
            ok = False
        print(f"{table.name:16s} src={len(rows):6d} 新增={inserted:6d} "
              f"跳过={skipped:6d} dst={dst_total:6d} [{status}]")

    # 显式 id 迁移后,PG SERIAL 序列仍停留在旧值,必须推进(仅对有 id 列的表)
    with dst.begin() as conn:
        for table in TABLES:
            if "id" not in table.c:
                continue
            seq = conn.execute(
                text("SELECT pg_get_serial_sequence(:t, 'id')"),
                {"t": table.name},
            ).scalar()
            if seq:
                conn.execute(text(
                    f"SELECT setval('{seq}', "
                    f"COALESCE((SELECT MAX(id) FROM {table.name}), 1))"
                ))
    print("序列已推进 (setval)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="SQLite 业务库 → IvorySQL/PostgreSQL 迁移")
    ap.add_argument("--src", required=True, help="源 DSN, 如 sqlite:///data/users.db")
    ap.add_argument("--dst", required=True, help="目标 DSN, 如 postgresql+psycopg://rag:rag@host:5432/rag_db")
    args = ap.parse_args()
    ok = migrate(args.src, args.dst)
    print("\n迁移" + ("完成" if ok else "存在不一致,请检查上面的 MISMATCH 行"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
