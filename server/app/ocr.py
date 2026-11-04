"""扫描件 OCR:整页渲染 → RapidOCR 全文转录。

定位:与视觉模型「插图理解」互补——
- 插图(流程图/示意图/盖章页)走视觉模型描述(vision.py);
- 扫描页(整页是一张文字图片、无文本层)走本模块全文转录,转录文字
  作为普通文本参与切分与索引,原页图像仍随 PDF 留存可溯源。

判定(is_scan_page):页文本层字符数 < SCAN_TEXT_MIN 且图片覆盖率
≥ SCAN_IMG_COVER。满足即整页渲染(200 DPI)进 RapidOCR;不满足维持
原文本流。OCR 引擎为 RapidOCR(PP-OCRv4,ONNX,CPU),模型随 pip 包
内置,离线可用;引擎不可用时回退原管线(扫描页退化为插图描述),
不阻断解析。
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from . import config

logger = logging.getLogger(__name__)

SCAN_TEXT_MIN = 40        # 页文本层少于此字数才考虑扫描页
SCAN_IMG_COVER = 0.5      # 图片面积占页面比例阈值
OCR_DPI = 200             # 整页渲染分辨率(清晰度/耗时平衡)

_ocr = None
_ocr_lock = threading.Lock()
_ocr_failed = False


def available() -> bool:
    """rapidocr 是否可导入(不触发模型加载)。"""
    if _ocr_failed:
        return False
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _engine():
    """懒加载单例;首次调用加载 ONNX 模型(约 1-2s)。"""
    global _ocr, _ocr_failed
    with _ocr_lock:
        if _ocr is None and not _ocr_failed:
            try:
                from rapidocr_onnxruntime import RapidOCR
                _ocr = RapidOCR()
            except Exception as e:  # 模型损坏/缺依赖:整功能降级,不炸解析
                _ocr_failed = True
                logger.warning("RapidOCR 初始化失败,扫描件 OCR 降级为插图描述: %s", e)
                return None
        return _ocr


def is_scan_page(page) -> bool:
    """判定一页是否为扫描页:文本层几乎没有 + 大图覆盖。"""
    try:
        text = page.get_text("text").strip()
        if len(text) >= SCAN_TEXT_MIN:
            return False
        infos = page.get_image_info()
        if not infos:
            return False
        page_area = abs(page.rect) or 1.0
        cover = 0.0
        import pymupdf
        for info in infos:
            r = pymupdf.Rect(info["bbox"]) & page.rect
            if not r.is_empty:
                cover += r.get_area()
        return cover / page_area >= SCAN_IMG_COVER
    except Exception:
        return False


def ocr_page(page) -> str:
    """整页渲染并转录;按检测框纵坐标排序拼接成行文本。失败返回 ""。"""
    eng = _engine()
    if eng is None:
        return ""
    try:
        pix = page.get_pixmap(dpi=OCR_DPI)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:                       # RGBA → RGB
            arr = arr[:, :, :3]
        result, _ = eng(arr)
        if not result:
            return ""
        lines = [r for r in result if r and len(r) >= 3 and float(r[2]) >= 0.4]
        lines.sort(key=lambda r: (round(float(r[0][0][1]) / 15), float(r[0][0][0])))
        return "\n".join(str(r[1]).strip() for r in lines if str(r[1]).strip())
    except Exception as e:
        logger.warning("页面 OCR 失败(跳过本页转录): %s", e)
        return ""


def enabled() -> bool:
    return bool(getattr(config, "SCAN_OCR", True))
