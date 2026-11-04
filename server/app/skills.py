"""技能(Skills):SKILL.md 指令包的加载、校验与管理(二期)。

设计要点:
- 技能 = 含 SKILL.md 的目录:YAML frontmatter(name/description)+ Markdown
  正文,本质是"提示词包"而非可执行代码——模型经 use_skill 工具按需加载
  正文并遵循,与 MCP(现成工具)互补:技能教模型"怎么做事",工具给模型
  "做事的手段";
- 双来源:server/skills/(内置示例,随代码分发,只读)与 DATA_DIR/skills/
  (管理员上传,可增删启停);内置名保留,上传目录禁止与之同名;
- 渐进式披露(progressive disclosure):系统提示只列 name + description,
  正文经 use_skill 按需拉取——几十个技能也不会撑爆系统提示;
- 安全边界:上传仅系统管理员;zip 校验(大小/条目数/路径穿越/SKILL.md
  必须存在);本环境只加载指令文本,技能目录里的脚本一律不执行;
- 启用状态存 settings.json 的 skills_disabled(名单制),全局开关
  skills_enabled 关闭时技能清单与 use_skill 同步下线。
"""
from __future__ import annotations

import io
import logging
import re
import shutil
import time
import zipfile
from pathlib import Path

from . import config as app_config

logger = logging.getLogger(__name__)

BUILTIN_DIR = app_config.BASE_DIR / "skills"      # 内置技能(随代码分发)
USER_DIR = app_config.DATA_DIR / "skills"         # 管理员上传的技能
USER_DIR.mkdir(parents=True, exist_ok=True)

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_SKILL_BYTES = 32 * 1024        # 单个 SKILL.md 上限(指令过长会稀释模型注意力)
MAX_ZIP_BYTES = 2 * 1024 * 1024    # 上传 zip 上限
MAX_ZIP_ENTRIES = 50               # zip 条目数上限
MAX_ZIP_TOTAL = 8 * 1024 * 1024    # zip 解压总量上限(防 zip 炸弹)
_LIST_TTL = 30                     # 清单缓存秒数(分类/系统提示每次请求都读)

_list_cache: tuple[float, list] | None = None


def invalidate() -> None:
    """上传/删除/启停后调用,清单缓存立即失效。"""
    global _list_cache
    _list_cache = None


def skills_on() -> bool:
    return bool(app_config.load_settings().get("skills_enabled", True))


# ------------------------------------------------------------ 解析 SKILL.md

def _parse(text: str) -> tuple[dict, str]:
    """解析 YAML frontmatter(手写极简解析器,只认 key: value 单层,不引
    pyyaml——frontmatter 里的值不允许复杂结构,引入依赖得不偿失)。"""
    lines = text.replace("\r\n", "\n").split("\n")
    meta: dict = {}
    body = text
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                meta = {}
                for ln in lines[1:i]:
                    if ":" in ln and not ln.startswith((" ", "\t", "#")):
                        k, v = ln.split(":", 1)
                        meta[k.strip()] = v.strip().strip("\"'")
                body = "\n".join(lines[i + 1:]).strip()
                break
            if i > 60:            # frontmatter 不可能这么长,防畸形文件扫全文
                break
    return meta, body


def _valid_skill_dir(d: Path) -> dict | None:
    f = d / "SKILL.md"
    if not d.is_dir() or not f.is_file():
        return None
    try:
        raw = f.read_bytes()
    except OSError:
        return None
    if len(raw) > MAX_SKILL_BYTES:
        logger.warning("技能 %s 的 SKILL.md 超过 %d 字节,已忽略", d.name, MAX_SKILL_BYTES)
        return None
    meta, body = _parse(raw.decode("utf-8", errors="replace"))
    name = str(meta.get("name") or d.name).strip()
    if not NAME_RE.fullmatch(name):
        logger.warning("技能目录 %s 的 name 非法(%r),已忽略", d.name, name)
        return None
    desc = str(meta.get("description") or "").strip()
    if not desc:
        logger.warning("技能 %s 缺少 description,已忽略", name)
        return None
    return {"name": name, "description": desc[:200], "body": body,
            "size": len(body), "dir": d}


def _scan(root: Path) -> list[dict]:
    out = []
    try:
        children = sorted(root.iterdir()) if root.is_dir() else []
    except OSError:
        return out
    for d in children:
        item = _valid_skill_dir(d)
        if item:
            out.append(item)
    return out


def list_skills(enabled_only: bool = False) -> list[dict]:
    """合并内置与上传技能(同名内置优先),附启用状态;带 30s 缓存。

    返回项不含 body(渐进式披露:清单只给 name/description/size)。
    """
    global _list_cache
    if _list_cache and _list_cache[0] > time.monotonic() and not enabled_only:
        return [dict(it) for it in _list_cache[1]]
    disabled = set(app_config.load_settings().get("skills_disabled") or [])
    merged: dict[str, dict] = {}
    for it in _scan(USER_DIR):
        it["builtin"] = False
        merged[it["name"]] = it
    for it in _scan(BUILTIN_DIR):           # 内置后写,同名覆盖用户目录
        it["builtin"] = True
        merged[it["name"]] = it
    items = []
    for it in sorted(merged.values(), key=lambda x: (not x["builtin"], x["name"])):
        it = {k: v for k, v in it.items() if k not in ("body", "dir")}
        it["enabled"] = it["name"] not in disabled
        items.append(it)
    _list_cache = (time.monotonic() + _LIST_TTL, items)
    if enabled_only:
        items = [it for it in items if it["enabled"]]
    return [dict(it) for it in items]


def get_skill(name: str) -> dict | None:
    """use_skill 工具调用:按名取正文(每次都读文件,技能本身体量小)。

    未启用/不存在返回 None——由调用方转成对模型友好的错误文本。
    """
    if not skills_on():
        return None
    disabled = set(app_config.load_settings().get("skills_disabled") or [])
    if name in disabled:
        return None
    for root in (BUILTIN_DIR, USER_DIR):    # 内置优先
        item = _valid_skill_dir(root / name)
        if item:
            return item
    return None


# ------------------------------------------------------------ 启停 / 删除

def set_enabled(name: str, enabled: bool) -> bool:
    s = app_config.load_settings()
    disabled = list(s.get("skills_disabled") or [])
    if enabled and name in disabled:
        disabled.remove(name)
    elif not enabled and name not in disabled:
        disabled.append(name)
    else:
        return any(it["name"] == name for it in list_skills())
    app_config.save_settings({"skills_disabled": disabled})
    invalidate()
    return True


def delete_skill(name: str) -> None:
    if not NAME_RE.fullmatch(name or ""):
        raise ValueError("技能名非法")
    if (BUILTIN_DIR / name / "SKILL.md").is_file():
        raise ValueError("内置技能不可删除(可在列表中停用)")
    target = USER_DIR / name
    if not target.is_dir():
        raise ValueError("技能不存在")
    shutil.rmtree(target)
    # 顺手从禁用名单清掉,避免名单残留指向不存在的技能
    s = app_config.load_settings()
    disabled = [n for n in (s.get("skills_disabled") or []) if n != name]
    app_config.save_settings({"skills_disabled": disabled})
    invalidate()


# ------------------------------------------------------------ 上传校验

def _builtin_names() -> set[str]:
    return {it["name"] for it in _scan(BUILTIN_DIR)}


def _check_name_available(name: str) -> None:
    if not NAME_RE.fullmatch(name or ""):
        raise ValueError("技能名需为小写字母/数字/中划线(1-64 位),如 weekly-report")
    if name in _builtin_names():
        raise ValueError(f"「{name}」与内置技能同名,请改名后上传")


def _write_skill(name: str, files: dict[str, bytes]) -> dict:
    dest = USER_DIR / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel, data in files.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    invalidate()
    item = _valid_skill_dir(dest)
    assert item is not None
    logger.info("技能已上传: %s(%d 个文件)", name, len(files))
    return {k: v for k, v in item.items() if k not in ("body", "dir")} | {"builtin": False, "enabled": True}


def save_upload(filename: str, data: bytes) -> dict:
    """上传入口:支持单个 .md 或含 SKILL.md 的 .zip;返回技能清单项。

    一切校验失败抛 ValueError,由 API 层转 400——校验逻辑不出本模块。
    """
    fname = (filename or "").lower()
    if fname.endswith(".md"):
        if len(data) > MAX_SKILL_BYTES:
            raise ValueError(f"SKILL.md 超过 {MAX_SKILL_BYTES // 1024}KB 上限")
        meta, _ = _parse(data.decode("utf-8", errors="replace"))
        name = str(meta.get("name") or "").strip()
        _check_name_available(name)
        return _write_skill(name, {"SKILL.md": data})
    if not fname.endswith(".zip"):
        raise ValueError("仅支持 .md 或 .zip 文件(zip 内需含 SKILL.md)")
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError(f"zip 超过 {MAX_ZIP_BYTES // 1024 // 1024}MB 上限")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("文件不是合法的 zip") from None
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_ENTRIES:
        raise ValueError(f"zip 内文件超过 {MAX_ZIP_ENTRIES} 个上限")
    total = 0
    files: dict[str, bytes] = {}
    skill_md_rel = None
    for info in infos:
        if info.is_dir():
            continue
        rel = info.filename.replace("\\", "/").lstrip("/")
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if not parts or any(p == ".." for p in parts):
            raise ValueError(f"zip 内含非法路径: {info.filename}")
        total += info.file_size
        if total > MAX_ZIP_TOTAL:
            raise ValueError("zip 解压总量超限")
        # SKILL.md 允许位于根目录或唯一的一层子目录里
        if parts[-1].lower() == "skill.md" and len(parts) <= 2:
            if skill_md_rel is not None:
                raise ValueError("zip 内存在多个 SKILL.md")
            skill_md_rel = "/".join(parts)
        if len(parts) > 3:
            raise ValueError(f"目录层级过深: {info.filename}")
        files["/".join(parts)] = zf.read(info)
    if skill_md_rel is None:
        raise ValueError("zip 内未找到 SKILL.md(位于根目录或一层子目录)")
    meta, _ = _parse(files[skill_md_rel].decode("utf-8", errors="replace"))
    name = str(meta.get("name") or "").strip()
    _check_name_available(name)
    # 规整:SKILL.md 统一放目标根目录,其余资源保持相对位置
    out_files = {"SKILL.md": files.pop(skill_md_rel)}
    prefix = skill_md_rel.rsplit("/", 1)[0] + "/" if "/" in skill_md_rel else ""
    for rel, blob in files.items():
        if prefix and rel.startswith(prefix):
            rel = rel[len(prefix):]
        out_files[rel] = blob
    return _write_skill(name, out_files)
