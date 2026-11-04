"""文档解析:文本流 + 表格提取 + 图片抽取(PDF)。

支持 PDF / DOCX / XLSX 三类:
- PDF:PyMuPDF 提取文本块、表格与图片。表格转为 Markdown 单独成块,不与正文
  混排强拆;清洗页眉(“第X部分……”行)与页码行、目录省略号噪声行;图片按
  阅读顺序以 [[IMG:id]] 占位标记注入文本流,字节存 images/ 目录——视觉模型
  描述在构建索引时生成(tasks.py),解析阶段零 LLM 调用。整页无文本层且被大图
  覆盖的扫描页走 OCR 全文转录(ocr.py,RapidOCR/PP-OCRv4,离线),转录文字
  作为普通文本进切分,页码记入 ParsedDoc.ocr_pages。
- DOCX:python-docx 按文档顺序遍历段落与表格(表格转 Markdown 块);图片直接跳过。
  DOCX 无固定分页,page 恒为 1(切分策略的页码回查退化为 1)。
- XLSX:openpyxl 逐工作表读取,每个非空 sheet 转一个 Markdown 表格块;
  正文为空字符串(表格即全部内容)。

输出统一的 ParsedDoc:整文档连续文本流 full_text + 页码映射 page_starts
+ 图片清单 images。
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf

from . import config, ocr as ocr_mod

_HEADER_RE = re.compile(r"^第[一二三四五六七八九十]+部分\S{0,20}$")
_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
_TOC_DOTS_RE = re.compile(r"[…·\.]{4,}")
_ROMAN_RE = re.compile(r"^[IVXL]{1,6}$")

# 图片占位标记(解析期):[[IMG:img_p7_2]];构建期被替换为 〔图片 …〕 描述块
IMG_MARK_RE = re.compile(r"\[\[IMG:([\w-]+)\]\]")
IMG_MIN_PX = 100          # 过滤装饰小图标(宽或高不足即跳过)
IMG_MAX_BYTES = 8 * 1024 * 1024
IMG_MAX_PER_DOC = 40      # 单文档图片处理上限(成本控制)


@dataclass
class TableBlock:
    page: int
    markdown: str
    rows: int
    cols: int


@dataclass
class ParsedDoc:
    name: str
    page_count: int
    full_text: str
    page_starts: list[int] = field(default_factory=list)  # 每页在 full_text 中的起始偏移
    tables: list[dict] = field(default_factory=list)      # TableBlock asdict
    images: list[dict] = field(default_factory=list)      # {img_id,page,width,height,sha1,file}
    ocr_pages: list[int] = field(default_factory=list)    # 扫描件 OCR 转录的页码(1-based)

    def page_of(self, offset: int) -> int:
        """由字符偏移查页码(1-based)。"""
        if not self.page_starts:
            return 1
        idx = bisect.bisect_right(self.page_starts, offset) - 1
        return max(1, min(self.page_count, idx + 1))


def _clean_lines(text: str, page_no: int) -> list[str]:
    """清洗单页文本:去页眉/页码/目录噪声。"""
    lines = [ln.strip() for ln in text.splitlines()]
    out: list[str] = []
    for i, ln in enumerate(lines):
        if not ln:
            continue
        # 页码行(独立成行的纯数字或罗马数字)
        if _PAGE_NUM_RE.match(ln) or _ROMAN_RE.match(ln):
            continue
        # 页眉:正文页首部的“第X部分XX规章制度”
        if i <= 1 and page_no > 4 and _HEADER_RE.match(ln):
            continue
        # 目录行的省略号引导
        if _TOC_DOTS_RE.search(ln):
            ln = _TOC_DOTS_RE.sub("", ln).strip()
            if not ln:
                continue
        out.append(ln)
    return out


def _table_to_markdown(tab) -> tuple[str, int, int]:
    """pymupdf 表格对象 → markdown;优先 to_markdown,失败则手工拼接。"""
    try:
        md = tab.to_markdown(clean=True)
        rows = tab.row_count
        cols = tab.col_count
        return md, rows, cols
    except Exception:
        pass
    data = tab.extract()
    if not data:
        return "", 0, 0
    def cell(x):
        return ("" if x is None else str(x)).replace("\n", " ").replace("|", "\\|").strip()
    header = [cell(c) for c in data[0]]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in data[1:]:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines), len(data), len(header)


def parse_pdf(pdf_path: Path, doc_name: str | None = None,
              images_dir: Path | None = None) -> ParsedDoc:
    doc = pymupdf.open(pdf_path)
    page_texts: list[str] = []
    tables: list[TableBlock] = []
    images: list[dict] = []
    seen_sha1: set[str] = set()
    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)
    ocr_pages: list[int] = []
    ocr_warned = False
    for pno in range(len(doc)):
        page = doc[pno]
        # 0) 扫描页判定:无文本层且大图覆盖 → 整页 OCR 转录为正文,
        #    跳过插图抽取(整页图不是插图)与表格检测(无文字层必空)
        if ocr_mod.enabled() and ocr_mod.is_scan_page(page):
            if ocr_mod.available():
                ocr_text = ocr_mod.ocr_page(page)
                if ocr_text:
                    ocr_pages.append(pno + 1)
                    page_texts.append("\n".join(_clean_lines(ocr_text, pno + 1)))
                    continue
            elif not ocr_warned:
                ocr_warned = True
                import logging
                logging.getLogger(__name__).warning(
                    "检测到扫描页但未安装 rapidocr,扫描内容将退化为插图描述")
        # 1) 表格区域
        table_rects: list[pymupdf.Rect] = []
        try:
            found = page.find_tables()
            for t in found.tables:
                md, rows, cols = _table_to_markdown(t)
                if md and rows >= 2:
                    tables.append(TableBlock(page=pno + 1, markdown=md, rows=rows, cols=cols))
                    table_rects.append(pymupdf.Rect(t.bbox))
        except Exception:
            pass
        # 2) 文本块与图片块按阅读顺序(y0)合并;剔除落入表格区域的文本块
        items: list[tuple[float, str]] = []
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, btext, *_ = b
            rect = pymupdf.Rect(x0, y0, x1, y1)
            in_table = False
            for tr in table_rects:
                inter = rect & tr
                if not inter.is_empty and inter.get_area() > 0.35 * rect.get_area():
                    in_table = True
                    break
            if not in_table:
                items.append((y0, btext))
        # 3) 图片:过滤小图标/超大图,按内容 sha1 去重(每页重复的 logo 只留首个),
        #    占位标记按纵向位置插入文本流——版式信息只有在这一层还完整
        if images_dir is not None and len(images) < IMG_MAX_PER_DOC:
            try:
                infos = page.get_image_info(xrefs=True)
            except Exception:
                infos = []
            for info in sorted(infos, key=lambda i: i["bbox"][1]):
                if len(images) >= IMG_MAX_PER_DOC:
                    break
                w, h = int(info.get("width", 0)), int(info.get("height", 0))
                if w < IMG_MIN_PX or h < IMG_MIN_PX:
                    continue
                try:
                    ext_bytes = doc.extract_image(info["xref"])
                    raw, ext = ext_bytes["image"], ext_bytes.get("ext", "png")
                except Exception:
                    continue
                if len(raw) > IMG_MAX_BYTES:
                    continue
                sha1 = hashlib.sha1(raw).hexdigest()[:16]
                if sha1 in seen_sha1:
                    continue                      # 重复图不再占位,避免刷屏
                seen_sha1.add(sha1)
                img_id = f"img_p{pno + 1}_{len(images) + 1}"
                fname = f"{img_id}.{ext}"
                try:
                    (images_dir / fname).write_bytes(raw)
                except OSError:
                    continue
                images.append({"img_id": img_id, "page": pno + 1, "width": w,
                               "height": h, "sha1": sha1, "file": f"images/{fname}"})
                items.append((info["bbox"][1], f"[[IMG:{img_id}]]"))
        items.sort(key=lambda x: x[0])
        cleaned = _clean_lines("\n".join(t for _, t in items), pno + 1)
        page_texts.append("\n".join(cleaned))
    doc.close()

    # 3) 拼接全文并记录页码映射
    full_parts: list[str] = []
    page_starts: list[int] = []
    offset = 0
    for t in page_texts:
        page_starts.append(offset)
        full_parts.append(t)
        offset += len(t) + 1
    parsed = ParsedDoc(
        name=doc_name or Path(pdf_path).name,
        page_count=len(page_texts),
        full_text="\n".join(full_parts),
        page_starts=page_starts,
        tables=[asdict(t) for t in tables],
        images=images,
        ocr_pages=ocr_pages,
    )
    return parsed


def _markdown_table(header: list, rows: list[list]) -> str:
    """单元格内容 → markdown 表格(通用,docx/xlsx 共用)。"""
    def cell(x):
        return ("" if x is None else str(x)).replace("\n", " ").replace("|", "\\|").strip()
    lines = ["| " + " | ".join(cell(c) for c in header) + " |",
             "|" + "|".join(["---"] * max(1, len(header))) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def parse_docx(path: Path, doc_name: str | None = None) -> ParsedDoc:
    """DOCX:按文档顺序输出段落(正文)与表格(markdown 块);图片跳过。"""
    import docx
    from docx.document import Document as _Doc
    from docx.oxml.ns import qn
    from docx.table import Table as _Table
    from docx.text.paragraph import Paragraph as _Para

    document = docx.Document(str(path))
    texts: list[str] = []
    tables: list[TableBlock] = []

    def _iter_block_items(parent: _Doc):
        for child in parent.element.body.iterchildren():
            if child.tag == qn("w:p"):
                yield _Para(child, parent)
            elif child.tag == qn("w:tbl"):
                yield _Table(child, parent)

    for block in _iter_block_items(document):
        if isinstance(block, _Para):
            t = block.text.strip()
            if t:
                texts.append(t)
        else:
            rows = [[c.text for c in r.cells] for r in block.rows]
            if len(rows) >= 2:
                md = _markdown_table(rows[0], rows[1:])
                if md:
                    tables.append(TableBlock(page=1, markdown=md,
                                             rows=len(rows), cols=len(rows[0])))
    return ParsedDoc(name=doc_name or Path(path).name, page_count=1,
                     full_text="\n".join(texts), page_starts=[0],
                     tables=[asdict(t) for t in tables])


def parse_xlsx(path: Path, doc_name: str | None = None) -> ParsedDoc:
    """XLSX:每个非空 sheet 转一个 markdown 表格块(read_only 流式,控制内存)。"""
    import openpyxl
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    tables: list[TableBlock] = []
    try:
        for ws in wb.worksheets:
            rows = [[c for c in row] for row in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(v is not None and str(v).strip() for v in r)]
            if len(rows) < 2:
                continue
            md = _markdown_table(list(rows[0]), rows[1:])
            if md:
                tables.append(TableBlock(page=1, markdown=md,
                                         rows=len(rows), cols=len(rows[0])))
    finally:
        wb.close()
    return ParsedDoc(name=doc_name or Path(path).name, page_count=1,
                     full_text="", page_starts=[0],
                     tables=[asdict(t) for t in tables])


def parse_document(path: Path, doc_name: str | None = None,
                   images_dir: Path | None = None) -> ParsedDoc:
    """按扩展名分发到具体解析器;不支持的类型抛 ValueError。
    images_dir 仅对 PDF 生效:传入才抽图,不传则只解析文本与表格。"""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path, doc_name, images_dir=images_dir)
    if ext == ".docx":
        return parse_docx(path, doc_name)
    if ext == ".xlsx":
        return parse_xlsx(path, doc_name)
    raise ValueError(f"不支持的文件类型: {ext}(仅支持 PDF/DOCX/XLSX 文本类文档)")


def apply_captions(doc: ParsedDoc, captions: dict[str, str]) -> ParsedDoc:
    """把 full_text 中的 [[IMG:id]] 占位标记替换为最终图片块(单行):
      有描述:〔图片 id〕描述:……〔/图片〕
      无描述:〔图片 id〕(保留位置信息,检索不到正文但至少知道这里有图)
    描述块是单行——这是切分层"原子保护"的基础(见 chunking.py)。
    替换改变文本长度,故按页处理并重算 page_starts,保证页码回查不失准。
    """
    def repl(m: re.Match) -> str:
        img_id = m.group(1)
        desc = (captions.get(img_id) or "").strip()
        if desc:
            return f"〔图片 {img_id}〕描述:{desc}〔/图片〕"
        return f"〔图片 {img_id}〕"

    if not doc.images:
        # 无图文档也要把可能残留的占位标记清掉(理论上不存在),直接返回
        return ParsedDoc(name=doc.name, page_count=doc.page_count,
                         full_text=IMG_MARK_RE.sub("", doc.full_text),
                         page_starts=list(doc.page_starts),
                         tables=doc.tables, images=doc.images,
                         ocr_pages=doc.ocr_pages)
    if not doc.page_starts:
        return ParsedDoc(name=doc.name, page_count=doc.page_count,
                         full_text=IMG_MARK_RE.sub(repl, doc.full_text),
                         page_starts=doc.page_starts,
                         tables=doc.tables, images=doc.images,
                         ocr_pages=doc.ocr_pages)
    # 按页切开 → 页内替换 → 重算偏移再拼回
    pages: list[str] = []
    starts = list(doc.page_starts) + [len(doc.full_text) + 1]
    for i in range(len(doc.page_starts)):
        seg = doc.full_text[starts[i]:starts[i + 1] - 1]   # -1 去掉页间拼接的 \n
        pages.append(IMG_MARK_RE.sub(repl, seg))
    new_starts: list[int] = []
    offset = 0
    for t in pages:
        new_starts.append(offset)
        offset += len(t) + 1
    return ParsedDoc(name=doc.name, page_count=doc.page_count,
                     full_text="\n".join(pages), page_starts=new_starts,
                     tables=doc.tables, images=doc.images,
                     ocr_pages=doc.ocr_pages)


def save_parsed(parsed: ParsedDoc, store_path: Path):
    store_path.write_text(
        json.dumps(asdict(parsed), ensure_ascii=False), "utf-8"
    )


def load_parsed(store_path: Path) -> ParsedDoc | None:
    if not store_path.exists():
        return None
    try:
        data = json.loads(store_path.read_text("utf-8"))
        return ParsedDoc(**data)
    except Exception:
        return None


def ensure_kb_document(kb_id: int) -> ParsedDoc | None:
    """确保指定知识库有可用文档:无缓存且是内置演示库(kb_id=1)时解析种子 PDF。"""
    store_path = config.kb_dir(kb_id) / "document.json"
    parsed = load_parsed(store_path)
    if parsed is not None:
        return parsed
    if kb_id == 1 and config.SEED_PDF.exists():
        parsed = parse_pdf(config.SEED_PDF, "员工规章制度手册",
                           images_dir=config.kb_dir(kb_id) / "images")
        save_parsed(parsed, store_path)
        return parsed
    return None
