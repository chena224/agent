"""图片描述(视觉模型):构建索引时把文档图片转成中文描述文本。

设计要点:
- 模型走 DeepSeek 官方 OpenAI 兼容接口的图片输入(deepseek-flash 系),
  与文本链路共用 Key/Base URL,视觉模型名单独可配(设置页 vision_model);
- 描述带上下文:把图片前后文段落 + 文档名一起给模型,描述才能用文档
  自己的术语(“请假审批流程”而不是“方框连菱形”),检索命中率全靠这一步;
- 描述输出强约束:单行、≤220 字、只说图中信息——它是检索语料不是文学作品;
- 缓存按图片 sha1 存 kb_dir/image_captions.json:重建索引不重复花钱;
- 失败/未配 Key 不阻断构建:该图无描述(占位块保留位置信息),进度里明示。
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path

from . import config as app_config
from .parser import IMG_MARK_RE, ParsedDoc

logger = logging.getLogger(__name__)

CAPTION_MAX_CHARS = 220
_CAPTION_TIMEOUT_S = 40

_PROMPT = """你是文档图片标注员。文档《{doc_name}》中有一张图片,请为它写检索用描述。

图片附近的正文(供你理解图片语境,描述时不要照抄):
{context}

要求:
1. 先判断图片类型(流程图/结构图/表格图/截图/照片/印章/其他),再描述内容;
2. 描述要用文档自己的术语,说明“这张图表达了什么”,1~3 句,不超过 180 字;
3. 表格图尽量转录关键行列;流程图写清起点、终点与主要环节;
4. 只说图中确实存在的信息,不要推测、不要编造;
5. 输出单行纯文本,不要换行、不要引号、不要“这是一张图”这类废话前缀。"""


def _captions_path(kb_id: int) -> Path:
    return app_config.kb_dir(kb_id) / "image_captions.json"


def load_captions(kb_id: int) -> dict[str, str]:
    """已生成的图片描述缓存:{img_id: 描述}。文件损坏视为空。"""
    try:
        data = json.loads(_captions_path(kb_id).read_text("utf-8"))
        return {str(k): str(v) for k, v in data.get("captions", {}).items()}
    except Exception:
        return {}


def _save_captions(kb_id: int, captions: dict[str, str], model: str) -> None:
    payload = {"model": model, "updated_at": int(time.time()), "captions": captions}
    try:
        _captions_path(kb_id).write_text(
            json.dumps(payload, ensure_ascii=False), "utf-8")
    except OSError as e:
        logger.warning("图片描述缓存写入失败 kb=%s: %s", kb_id, e)


def _context_around(doc: ParsedDoc, img_id: str, span: int = 260) -> str:
    """取占位标记前后各 span 字符的正文,作为视觉模型的语境。"""
    pos = -1
    for mm in IMG_MARK_RE.finditer(doc.full_text):
        if mm.group(1) == img_id:
            pos = mm.start()
            break
    if pos < 0:
        return "(无上下文)"
    s = max(0, pos - span)
    e = min(len(doc.full_text), pos + span)
    ctx = doc.full_text[s:e]
    return IMG_MARK_RE.sub("[另一张图]", ctx).strip() or "(无上下文)"


def _sanitize(text: str) -> str:
    """描述净化:单行、去标记字符、限量——它是检索语料,格式必须可预测。"""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    t = t.replace("〔", "(").replace("〕", ")").replace("[[IMG:", "[IMG:")
    return t[:CAPTION_MAX_CHARS]


def caption_images(kb_id: int, doc: ParsedDoc,
                   progress_cb=None) -> tuple[dict[str, str], str]:
    """为文档中还没有缓存描述的图片生成描述。

    返回 ({img_id: 描述}, 状态说明)。状态说明用于构建进度展示:
    未配置 Key / 全部失败 / 部分成功都有明确文案,绝不静默。
    """
    existing = load_captions(kb_id)
    todo = [im for im in doc.images if not existing.get(im["img_id"])]
    if not todo:
        return existing, ""

    s = app_config.load_settings()
    api_key = s.get("deepseek_api_key") or s.get("deepseek_backup_api_key") or ""
    if not api_key:
        logger.warning("知识库 %s 有 %d 张图片待描述,但未配置 API Key,跳过",
                       kb_id, len(todo))
        return existing, f"{len(todo)} 张图片未生成描述(未配置 API Key)"

    from langchain_openai import ChatOpenAI   # 延迟导入:纯文本构建不触达
    model = s.get("vision_model") or "deepseek-flash"
    llm = ChatOpenAI(model=model, api_key=api_key,
                     base_url=s.get("deepseek_base_url"),
                     max_tokens=256, temperature=0, timeout=_CAPTION_TIMEOUT_S)

    captions = dict(existing)
    ok, fail = 0, 0
    for i, im in enumerate(todo):
        if progress_cb:
            progress_cb(i + 1, len(todo))
        img_path = app_config.kb_dir(kb_id) / im["file"]
        try:
            raw = img_path.read_bytes()
            b64 = base64.b64encode(raw).decode()
            ext = img_path.suffix.lstrip(".").lower() or "png"
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            prompt = _PROMPT.format(doc_name=doc.name,
                                    context=_context_around(doc, im["img_id"]))
            resp = llm.invoke([{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }])
            desc = _sanitize(getattr(resp, "content", ""))
            if desc:
                captions[im["img_id"]] = desc
                ok += 1
            else:
                fail += 1
                logger.warning("图片描述为空 kb=%s img=%s", kb_id, im["img_id"])
        except Exception as e:
            fail += 1
            logger.warning("图片描述失败 kb=%s img=%s: %s", kb_id, im["img_id"], e)
        # 每处理 5 张落一次盘:中途崩溃不丢已完成的描述
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            _save_captions(kb_id, captions, model)

    note = ""
    if fail:
        note = f"{fail}/{len(todo)} 张图片描述失败(重建索引可重试)"
    return captions, note
