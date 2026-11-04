#!/usr/bin/env python3
"""双库一致性检查:业务库(IvorySQL/SQLite) ↔ 向量库(Milvus)

对账口径(路线二 四.3):
- documents 台账是文档级删除/审计的锚点,Milvus 中每个分块带 doc_id 字段;
- 台账有文档而 Milvus 无对应向量(且该库已构建) → 缺失向量;
- Milvus 有向量而台账无对应文档 → 孤儿向量;
- doc_id 为空串的分块是旧版(无 doc_id 时代)遗留,只按总数对账,不算孤儿。

用法(在 server/ 目录下,环境变量与线上服务一致):
    PYTHONPATH=. python3 scripts/check_consistency.py           # 检查全部知识库
    PYTHONPATH=. python3 scripts/check_consistency.py --kb 3    # 只查一个库
退出码:0 一致 / 1 存在不一致。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # server/

from app import db, store  # noqa: E402


def check_kb(vs: store.VectorStore, kb: dict) -> list[str]:
    problems: list[str] = []
    kb_id, name = kb["id"], kb["name"]
    docs = db.list_documents(kb_id)
    ledger_ids = {str(d["id"]) for d in docs}
    try:
        milvus_counts = vs.doc_chunk_counts(kb_id)      # {doc_id: n}
    except Exception as exc:                             # 集合不存在 = 尚未构建
        if kb.get("chunk_count"):
            problems.append(f"KB#{kb_id}[{name}] 台账显示 {kb['chunk_count']} 块,但 Milvus 集合读取失败: {exc}")
        return problems
    legacy = milvus_counts.get("", 0)                    # 旧版无 doc_id 分块
    vector_ids = {k for k in milvus_counts if k}

    missing = ledger_ids - vector_ids
    orphan = vector_ids - ledger_ids
    if missing and kb.get("chunk_count"):
        names = {str(d["id"]): d["filename"] for d in docs}
        for d in sorted(missing):
            problems.append(f"KB#{kb_id}[{name}] 缺失向量: 文档#{d} {names.get(d, '?')} 在台账存在但 Milvus 无分块")
    for d in sorted(orphan):
        problems.append(f"KB#{kb_id}[{name}] 孤儿向量: doc_id={d} 在 Milvus 有 {milvus_counts[d]} 块但台账无此文档")

    total_vec = sum(milvus_counts.values())
    if legacy:
        print(f"KB#{kb_id}[{name}] 提示: {legacy} 块为旧版无 doc_id 分块,建议重建索引以获得文档级删除能力")
    if kb.get("chunk_count") and total_vec != kb["chunk_count"]:
        problems.append(f"KB#{kb_id}[{name}] 总数不符: 台账 chunk_count={kb['chunk_count']} vs Milvus 实际 {total_vec}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="业务库 ↔ Milvus 双库一致性检查")
    ap.add_argument("--kb", type=int, default=0, help="只检查指定知识库 id")
    args = ap.parse_args()

    db.init_db()
    vs = store.VectorStore.get()
    try:    # Milvus Lite 单进程独占:服务运行时本脚本连不上,先探活给明确提示
        vs.client.list_collections()
    except Exception as exc:
        print(f"Milvus 不可达({exc.__class__.__name__})。standalone 档请先停服再对账;"
              f"cluster 档请检查 MILVUS_URI。", file=sys.stderr)
        sys.exit(2)
    kbs = db.list_kbs()
    if args.kb:
        kbs = [k for k in kbs if k["id"] == args.kb]
        if not kbs:
            print(f"知识库 #{args.kb} 不存在", file=sys.stderr)
            sys.exit(1)

    problems: list[str] = []
    for kb in kbs:
        problems.extend(check_kb(vs, kb))

    if problems:
        print("\n发现不一致:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print(f"一致: {len(kbs)} 个知识库台账与 Milvus 对齐")
    sys.exit(0)


if __name__ == "__main__":
    main()
