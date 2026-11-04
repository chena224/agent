"""切分策略:重叠窗口 / 递归字符 / Token / 条目结构 / 语义 / 父子块。

覆盖 LangChain 主流切分方式:
- fixed        CharacterTextSplitter 式固定长度滑窗(重叠窗口)
- recursive    RecursiveCharacterTextSplitter 递归字符切分
- token        TokenTextSplitter 按 Token 数切分(tiktoken cl100k_base)
- structural   自定义:按“第X条”条目边界切分并携带章节上下文
- semantic     语义切分:句子向量余弦距离探测拐点
- parent_child 父子块(ParentDocumentRetriever 模式):父块留上下文,子块做检索

所有策略共享同一后处理:
- 由 page_starts 回查每块页码范围
- 表格块(TableBlock)整体成块,不参与任何切分,保证表格语义完整
- 图片块(〔图片 id〕描述…〔/图片〕,单行)原子保护:任何策略都不会把它
  从中间切开;fixed/token 这类滑窗策略把它独立成块,段落类策略让它随文留存
- 最终按页码排序输出,附带统一元数据
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parser import ParsedDoc

# 匹配“第X条”起始(支持中文数字与阿拉伯数字)
_ARTICLE_RE = re.compile(r"(?=第[一二三四五六七八九十百千零〇两\d]{1,7}条)")
_CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十\d]{1,4}章\S{0,30})$")
_SECTION_RE = re.compile(r"^(第[一二三四五六七八九十\d]{1,4}节\S{0,30})$")
_SENT_RE = re.compile(r"(?<=[。！？；!?;])\s*|\n+")

# 图片块(单行):〔图片 id〕描述:…〔/图片〕 或裸标记 〔图片 id〕
_IMG_BLOCK_RE = re.compile(r"〔图片 ([\w-]+)〕(?:[^\n]*)")


def image_ids_of(text: str) -> list[str]:
    """一块文本里包含的全部图片 id(供溯源回链原图)。"""
    return [m.group(1) for m in _IMG_BLOCK_RE.finditer(text)]


def _split_img_units(text: str) -> list[tuple[bool, str, int]]:
    """把文本切成 (是否图片块, 内容, 绝对偏移) 交替序列,内容无损拼接。
    fixed/token 这类不看段落的滑窗策略靠它保证图片块原子性。"""
    out: list[tuple[bool, str, int]] = []
    cursor = 0
    for m in _IMG_BLOCK_RE.finditer(text):
        if m.start() > cursor:
            out.append((False, text[cursor:m.start()], cursor))
        out.append((True, m.group(0), m.start()))
        cursor = m.end()
    if cursor < len(text):
        out.append((False, text[cursor:], cursor))
    return out


def _sent_units(text: str) -> list[str]:
    """句子切分的图片安全版:图片块整体作为一个“句子”单元,其余文本按
    句读拆分——防止描述里的句号把 〔图片…〕 拦腰截断。"""
    units: list[str] = []
    for is_img, content, _ in _split_img_units(text):
        if is_img:
            units.append(content)
        else:
            units.extend(s for s in _SENT_RE.split(content) if s and s.strip())
    return units


@dataclass
class Chunk:
    text: str
    kind: str = "text"            # text | table | image
    page_start: int = 1
    page_end: int = 1
    chapter: str = ""
    article: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self, idx: int) -> dict:
        return {
            "id": idx,
            "text": self.text,
            "kind": self.kind,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chapter": self.chapter,
            "article": self.article,
            "length": len(self.text),
            "parent_id": self.meta.get("parent_id"),
            "parent_text": self.meta.get("parent_text", ""),
            "image_ids": self.meta.get("image_ids", []),
        }


STRATEGIES = {
    "fixed": {"name": "重叠窗口切分", "desc": "CharacterTextSplitter 式固定长度滑窗,重叠区长度可调(填 0 即为直接截断),最简单粗暴的基线策略"},
    "recursive": {"name": "递归字符切分", "desc": "RecursiveCharacterTextSplitter:按段落→换行→句号→分号优先级递归切分,尽量保持语义边界"},
    "token": {"name": "Token 切分", "desc": "TokenTextSplitter:按 Token 数直接截断、无重叠(tiktoken cl100k_base),块大小与大模型上下文窗口直接对齐"},
    "structural": {"name": "条目结构切分", "desc": "按“第X条”条目边界切分并携带章节上下文,法规类文档的最优策略"},
    "semantic": {"name": "语义切分", "desc": "基于句子向量余弦距离探测语义拐点,语义突变处断开成块"},
    "parent_child": {"name": "父子块切分", "desc": "ParentDocumentRetriever 模式:父块保留完整上下文,子块参与向量检索,命中后回取父块喂给大模型"},
}


# ---------------------------------------------------------------- 基础工具

def _attach_pages(chunks: list[tuple[str, int, dict]], doc: ParsedDoc) -> list[Chunk]:
    """(text, start_offset, meta) → Chunk,回填页码;图片块识别与 image_ids 落元数据。"""
    out: list[Chunk] = []
    for text, start, meta in chunks:
        p1 = doc.page_of(start)
        p2 = doc.page_of(start + max(0, len(text) - 1))
        ids = image_ids_of(text)
        if ids:
            meta = {**meta, "image_ids": ids}
        # 整块就是一个图片块 → kind=image(预览/引用可区分展示)
        stripped = text.strip()
        kind = meta.get("kind") or (
            "image" if ids and _IMG_BLOCK_RE.fullmatch(stripped) else "text")
        meta.pop("kind", None)
        out.append(Chunk(
            text=stripped,
            kind=kind,
            page_start=p1, page_end=p2,
            chapter=meta.get("chapter", ""),
            article=meta.get("article", ""),
            meta=meta,
        ))
    return [c for c in out if c.text]


def _with_tables(chunks: list[Chunk], doc: ParsedDoc) -> list[Chunk]:
    """表格整体成块,按页码插入正文中相应位置。"""
    tbl_chunks = [
        Chunk(text=t["markdown"], kind="table", page_start=t["page"], page_end=t["page"],
              meta={"rows": t["rows"], "cols": t["cols"]})
        for t in doc.tables
    ]
    merged = chunks + tbl_chunks
    merged.sort(key=lambda c: (c.page_start, 0 if c.kind == "text" else 1))
    return merged


def _window_spans(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    spans = []
    step = max(1, size - overlap)
    start = 0
    while start < n:
        end = min(n, start + size)
        spans.append((start, end))
        if end >= n:
            break
        start += step
    return spans


# ---------------------------------------------------------------- 策略实现

def chunk_fixed(doc: ParsedDoc, chunk_size: int = 500, overlap: int = 50, **_) -> list[Chunk]:
    text = doc.full_text
    if "〔图片" not in text:
        spans = _window_spans(len(text), chunk_size, overlap)
        return _attach_pages([(text[s:e], s, {}) for s, e in spans], doc)
    # 含图片块:滑窗只作用于纯文本段,图片块独立成块,保证原子性
    chunks: list[tuple[str, int, dict]] = []
    for is_img, seg, base in _split_img_units(text):
        if is_img:
            chunks.append((seg, base, {}))
            continue
        for s, e in _window_spans(len(seg), chunk_size, overlap):
            chunks.append((seg[s:e], base + s, {}))
    return _attach_pages(chunks, doc)


def chunk_recursive(doc: ParsedDoc, chunk_size: int = 500, overlap: int = 50, **_) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "；", "，", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    # 手动定位每个分段在原文中的位置(顺序扫描)
    chunks: list[tuple[str, int, dict]] = []
    cursor = 0
    for piece in splitter.split_text(doc.full_text):
        idx = doc.full_text.find(piece[:40], cursor)
        if idx < 0:
            idx = cursor
        chunks.append((piece, idx, {}))
        cursor = idx + max(1, len(piece) - overlap)
    return _attach_pages(chunks, doc)


def _locate(pieces: list[str], full_text: str, overlap: int) -> list[tuple[str, int, dict]]:
    """顺序扫描定位每个分段在原文中的偏移。"""
    chunks: list[tuple[str, int, dict]] = []
    cursor = 0
    for piece in pieces:
        idx = full_text.find(piece[:40], cursor)
        if idx < 0:
            idx = cursor
        chunks.append((piece, idx, {}))
        cursor = idx + max(1, len(piece) - overlap)
    return chunks


def chunk_token(doc: ParsedDoc, chunk_size: int = 400, **_) -> list[Chunk]:
    """Token 切分:TokenTextSplitter(tiktoken cl100k_base 编码),零重叠直接截断。

    与 fixed(重叠窗口)刻意拉开差异:需要重叠语义的选 fixed(重叠区可调),
    需要与模型上下文窗口精确对齐的选本策略,不再暴露重叠参数。
    """
    from langchain_text_splitters import TokenTextSplitter
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    text = doc.full_text
    if "〔图片" not in text:
        return _attach_pages(_locate(splitter.split_text(text), text, 0), doc)
    # 含图片块:逐段切分,图片块独立成块(tiktoken 内部刀口不可控,切段隔离)
    chunks: list[tuple[str, int, dict]] = []
    for is_img, seg, base in _split_img_units(text):
        if is_img:
            chunks.append((seg, base, {}))
        else:
            for piece, off, meta in _locate(splitter.split_text(seg), seg, 0):
                chunks.append((piece, base + off, meta))
    return _attach_pages(chunks, doc)


def chunk_parent_child(doc: ParsedDoc, parent_size: int = 1200, parent_overlap: int = 100,
                       child_size: int = 300, child_overlap: int = 40, **_) -> list[Chunk]:
    """父子块切分(ParentDocumentRetriever 模式)。

    - 父块:递归字符切分,保留完整段落上下文,不直接参与检索;
    - 子块:在父块内部再递归细分,子块向量化参与检索;
    - 每个子块 meta 携带 parent_id / parent_text,命中后由代理层回取父块组装上下文。
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "；", "，", ""],
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "；", "，", ""],
        chunk_size=child_size,
        chunk_overlap=child_overlap,
    )
    full = doc.full_text
    chunks: list[tuple[str, int, dict]] = []
    cursor = 0
    parent_id = 0
    for parent in parent_splitter.split_text(full):
        p_idx = full.find(parent[:40], cursor)
        if p_idx < 0:
            p_idx = cursor
        cursor = p_idx + max(1, len(parent) - parent_overlap)
        child_cursor = p_idx
        for child in child_splitter.split_text(parent):
            c_idx = full.find(child[:40], child_cursor)
            if c_idx < 0:
                c_idx = child_cursor
            child_cursor = c_idx + max(1, len(child) - child_overlap)
            chunks.append((child, c_idx, {
                "parent_id": parent_id,
                "parent_text": parent,
                "child": True,
            }))
        parent_id += 1
    return _attach_pages(chunks, doc)


def chunk_structural(doc: ParsedDoc, max_len: int = 900, **_) -> list[Chunk]:
    """按“第X条”切分;超长条按句子二次切分;章/节标题作为上下文前缀。"""
    text = doc.full_text
    # 记录每个“第X章”标题及其位置
    chapters: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        m = _CHAPTER_RE.match(line.strip())
        if m:
            chapters.append((offset, m.group(1)))
        offset += len(line)

    def chapter_of(pos: int) -> str:
        cur = ""
        for cpos, title in chapters:
            if cpos <= pos:
                cur = title
            else:
                break
        return cur

    pieces = _ARTICLE_RE.split(text)
    chunks: list[tuple[str, int, dict]] = []
    cursor = 0
    for piece in pieces:
        idx = text.find(piece[:20], cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(piece)
        piece = piece.strip()
        if not piece:
            continue
        m = re.match(r"^(第[一二三四五六七八九十百千零〇两\d]{1,7}条)", piece)
        article = m.group(1) if m else ""
        chapter = chapter_of(idx)
        header = f"【{chapter}】" if chapter else ""
        if len(piece) <= max_len:
            chunks.append((header + piece, idx, {"chapter": chapter, "article": article}))
        else:
            # 超长条目按句子二次细分,后续分块携带“第X条(续)”上下文
            prefix = f"{header}{article}(续)" if article else header
            for j, s in enumerate(_sub_split(piece, max_len)):
                tag = header + s if j == 0 else prefix + s
                chunks.append((tag, idx, {"chapter": chapter, "article": article}))
    return _attach_pages(chunks, doc)


def _sub_split(text: str, max_len: int) -> list[str]:
    sents = _sent_units(text)   # 图片块整体作为单元,不被句读截断
    out, buf = [], ""
    for s in sents:
        if len(buf) + len(s) > max_len and buf:
            out.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        out.append(buf)
    return out or [text[:max_len]]


def chunk_semantic(doc: ParsedDoc, embedder, percentile: float = 90.0,
                   max_len: int = 600, min_len: int = 60, **_) -> list[Chunk]:
    """语义切分:句子级向量余弦距离超过分位阈值处断开。"""
    import numpy as np

    text = doc.full_text
    sents: list[tuple[str, int]] = []  # (句子/图片块, 起始偏移)
    cursor = 0
    for piece in _sent_units(text):
        idx = text.find(piece, cursor)
        if idx < 0:
            idx = cursor
        sents.append((piece.strip(), idx))
        cursor = idx + len(piece)
    if len(sents) < 2:
        return _attach_pages([(text, 0, {})], doc)

    vecs = embedder.embed_texts([s for s, _ in sents], batch_size=64)
    mat = np.asarray(vecs, dtype=np.float32)
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    mat = mat / norm
    sims = np.sum(mat[:-1] * mat[1:], axis=1)
    dists = 1.0 - sims
    threshold = float(np.percentile(dists, percentile))

    chunks: list[tuple[str, int, dict]] = []
    buf, buf_start = "", sents[0][1]
    for i, (sent, off) in enumerate(sents):
        buf += sent
        is_last = i == len(sents) - 1
        boundary = (not is_last) and (dists[i] > threshold or len(buf) >= max_len)
        if boundary or is_last:
            if len(buf) < min_len and chunks:
                # 过短的尾巴并入前一块
                ptext, pstart, pmeta = chunks[-1]
                chunks[-1] = (ptext + buf, pstart, pmeta)
            else:
                chunks.append((buf, buf_start, {}))
            if not is_last:
                buf, buf_start = "", sents[i + 1][1]
    return _attach_pages(chunks, doc)


# ---------------------------------------------------------------- 统一入口

def run_strategy(strategy: str, doc: ParsedDoc, embedder=None, params: dict | None = None) -> list[Chunk]:
    params = params or {}
    if strategy == "fixed":
        chunks = chunk_fixed(doc, **params)
    elif strategy == "recursive":
        chunks = chunk_recursive(doc, **params)
    elif strategy == "token":
        chunks = chunk_token(doc, **params)
    elif strategy == "structural":
        chunks = chunk_structural(doc, **params)
    elif strategy == "parent_child":
        chunks = chunk_parent_child(doc, **params)
    elif strategy == "semantic":
        if embedder is None:
            raise ValueError("语义切分需要向量模型支持")
        chunks = chunk_semantic(doc, embedder=embedder, **params)
    else:
        raise ValueError(f"未知切分策略: {strategy}")
    return _with_tables(chunks, doc)


def chunk_stats(chunks: list[Chunk]) -> dict:
    lens = [len(c.text) for c in chunks]
    return {
        "total": len(chunks),
        "text_chunks": sum(1 for c in chunks if c.kind == "text"),
        "table_chunks": sum(1 for c in chunks if c.kind == "table"),
        "image_chunks": sum(1 for c in chunks if c.kind == "image"),
        "avg_len": round(sum(lens) / max(1, len(lens)), 1),
        "max_len": max(lens) if lens else 0,
        "min_len": min(lens) if lens else 0,
    }
