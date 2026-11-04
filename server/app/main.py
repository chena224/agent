"""FastAPI 主入口:多知识库智能问答平台 API + 前端静态托管。

v2 变更:
- 登录认证(账号口令 + Bearer 令牌),RBAC 三角色权限控制(规格书 1.7)
- 多知识库:独立 Milvus 集合 kb_{id},按用户/部门授权,访问隔离
- 意图识别埋点;统计分用户/知识库/意图维度

接口一览:
  POST   /api/auth/login                 登录
  POST   /api/auth/logout                登出
  GET    /api/auth/me                    当前用户
  GET    /api/admin/users                用户列表(系统管理员)
  POST   /api/admin/users                新建用户(系统管理员)
  PATCH  /api/admin/users/{id}           修改用户(角色/部门/状态/重置口令)
  GET    /api/admin/departments          部门列表
  GET    /api/kbs                        可见知识库列表(含权限标记)
  POST   /api/kbs                        新建知识库(管理员)
  GET    /api/kbs/{kb_id}                知识库详情(文档/索引/策略/授权)
  DELETE /api/kbs/{kb_id}                删除知识库
  POST   /api/kbs/{kb_id}/upload         上传 PDF 并解析
  POST   /api/kbs/{kb_id}/chunks/preview 按策略预览分段
  POST   /api/kbs/{kb_id}/index/build    构建向量索引(后台任务)
  GET    /api/kbs/{kb_id}/index/progress 构建进度
  GET    /api/kbs/{kb_id}/index/status   索引状态
  DELETE /api/kbs/{kb_id}/index          清空索引
  GET    /api/kbs/{kb_id}/grants         授权列表
  POST   /api/kbs/{kb_id}/grants         新增授权(按用户/部门)
  DELETE /api/kbs/{kb_id}/grants/{gid}   撤销授权
  POST   /api/kbs/{kb_id}/retrieve       仅检索(调试)
  POST   /api/chat                       智能问答(SSE 流式 / 一次性)
  GET    /api/chat/sessions?kb_id=       会话列表(当前用户)
  GET    /api/chat/history/{kb_id}/{sid} 会话历史
  DELETE /api/chat/sessions/{kb_id}/{sid}删除会话
  GET    /api/stats/summary|timeseries|recent|intents|kbs   埋点统计(管理员)
  GET    /api/settings / POST /api/settings / POST /api/settings/test(系统管理员)
  GET    /api/health                     健康与组件状态
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile

# 应用日志:uvicorn 下接管 root logger,保证 router/agent/retrieval 的
# 配置告警与分支错误可见(语义路由命中、意图解析失败、检索分支故障等)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import analytics, cache, config, connectors as connectors_mod, db, tasks
from . import skills as skills_mod
from .agent import INTENT_LABELS, RAGAgent, thread_id_of
from .chunking import STRATEGIES, chunk_stats, run_strategy
from .embeddings import Embedder
from .router import SemanticRouter
from .parser import ensure_kb_document, load_parsed, parse_document, save_parsed
from .store import VectorStore

app = FastAPI(title="知识库智能体平台")
# CORS:默认 *(前后端同源托管,仅开发用);生产经 CORS_ORIGINS 收敛到具体来源(路线二 七.3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS or ["*"],
    allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """/api/chat 按用户固定窗口限流(路线二 七.2);多实例经 Redis 共享计数。"""
    if request.url.path == "/api/chat":
        user = None
        try:
            user = db.user_by_token(_token_of(request))
        except Exception:
            pass
        if user and not cache.rate_limit_ok(f"chat:{user['id']}",
                                            config.RATE_LIMIT_CHAT_PER_MIN, 60):
            return JSONResponse({"detail": "请求过于频繁,请稍后再试"}, status_code=429)
    return await call_next(request)


# ------------------------------------------------------------------ 认证/权限

def _token_of(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token", "")


def current_user(request: Request) -> dict:
    user = db.user_by_token(_token_of(request))
    if not user:
        raise HTTPException(401, "请先登录")
    return user


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "仅系统管理员可执行该操作")
    return user


def require_manager(user: dict = Depends(current_user)) -> dict:
    if user["role"] not in ("admin", "workspace_admin"):
        raise HTTPException(403, "仅管理员可执行该操作")
    return user


def _get_kb_or_404(kb_id: int) -> dict:
    kb = db.get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "知识库不存在")
    return kb


def _access_kb(user: dict, kb_id: int) -> dict:
    kb = _get_kb_or_404(kb_id)
    if not db.can_access_kb(user, kb):
        raise HTTPException(403, "没有该知识库的访问权限,请联系管理员授权")
    return kb


def _manage_kb(user: dict, kb_id: int) -> dict:
    kb = _get_kb_or_404(kb_id)
    if not db.can_manage_kb(user, kb):
        raise HTTPException(403, "没有该知识库的管理权限")
    return kb


# ------------------------------------------------------------------ 工具

# 切分缓存与键规则在 tasks.py(构建任务与预览共用),这里做薄别名保持端点简洁
_chunk_cache_path = tasks._chunk_cache_path
_chunk_cache_load = tasks.chunk_cache_load
_chunk_cache_save = tasks.chunk_cache_save
_cache_key = tasks.chunk_cache_key


def _require_doc(kb_id: int):
    doc = load_parsed(config.kb_dir(kb_id) / "document.json") or ensure_kb_document(kb_id)
    if doc is None:
        raise HTTPException(400, "该知识库尚未加载文档,请先上传 PDF")
    return doc


def _kb_with_status(user: dict, kb: dict) -> dict:
    item = dict(kb)
    item["can_manage"] = db.can_manage_kb(user, kb)
    item["index"] = VectorStore.get().status(kb["id"])
    item["has_document"] = (config.kb_dir(kb["id"]) / "document.json").exists()
    owner = db.get_user(kb["owner_id"])
    item["owner_name"] = owner["display_name"] if owner else ""
    return item


# ------------------------------------------------------------------ 认证接口

class LoginIn(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginIn):
    user = db.authenticate(body.username.strip(), body.password)
    if not user:
        raise HTTPException(401, "用户名或密码错误,或账号已停用")
    token = db.create_session(user["id"])
    return {"token": token, "user": user}


@app.post("/api/auth/logout")
def logout(request: Request):
    db.revoke_session(_token_of(request))
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return {"user": user}


# ------------------------------------------------------------------ 用户管理(系统管理员)

@app.get("/api/admin/users")
def admin_users(_: dict = Depends(require_admin)):
    return {"users": db.list_users(), "roles": db.ROLES, "departments": db.DEPARTMENTS}


class UserIn(BaseModel):
    username: str
    password: str
    display_name: str = ""
    department: str = ""
    role: str = "user"


@app.post("/api/admin/users")
def admin_create_user(body: UserIn, _: dict = Depends(require_admin)):
    if not body.username.strip() or not body.password:
        raise HTTPException(400, "用户名与口令不能为空")
    try:
        return {"user": db.create_user(body.username.strip(), body.password,
                                       body.display_name, body.department, body.role)}
    except ValueError as e:
        raise HTTPException(400, str(e))


class UserPatch(BaseModel):
    display_name: str | None = None
    department: str | None = None
    role: str | None = None
    status: int | None = None
    password: str | None = None


@app.patch("/api/admin/users/{user_id}")
def admin_update_user(user_id: int, body: UserPatch, admin: dict = Depends(require_admin)):
    if user_id == admin["id"] and (body.status == 0 or (body.role and body.role != "admin")):
        raise HTTPException(400, "不能停用或降级当前登录的管理员账号")
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    if patch.get("role") and patch["role"] not in db.ROLES:
        raise HTTPException(400, "非法角色")
    user = db.update_user(user_id, patch)
    if not user:
        raise HTTPException(404, "用户不存在")
    return {"user": user}


@app.get("/api/admin/departments")
def admin_departments(_: dict = Depends(require_manager)):
    depts = set(db.DEPARTMENTS)
    for u in db.list_users():
        if u["department"]:
            depts.add(u["department"])
    return {"departments": sorted(depts)}


# ------------------------------------------------------------------ 知识库

@app.get("/api/kbs")
def list_kbs(user: dict = Depends(current_user)):
    kbs = [_kb_with_status(user, kb) for kb in db.accessible_kbs(user)]
    return {"kbs": kbs}


class KBIn(BaseModel):
    name: str
    description: str = ""
    visibility: str = "private"


@app.post("/api/kbs")
def create_kb(body: KBIn, user: dict = Depends(require_manager)):
    if not body.name.strip():
        raise HTTPException(400, "知识库名称不能为空")
    kb = db.create_kb(body.name.strip(), body.description, body.visibility, user["id"])
    return {"kb": kb}


@app.get("/api/kbs/{kb_id}")
def kb_detail(kb_id: int, user: dict = Depends(current_user)):
    kb = _access_kb(user, kb_id)
    item = _kb_with_status(user, kb)
    doc = load_parsed(config.kb_dir(kb_id) / "document.json") or ensure_kb_document(kb_id)
    if doc:
        item["document"] = {
            "name": doc.name, "page_count": doc.page_count,
            "text_len": len(doc.full_text), "table_count": len(doc.tables),
            "tables": [{"page": t["page"], "rows": t["rows"], "cols": t["cols"]}
                       for t in doc.tables],
            "images": [{"img_id": i["img_id"], "page": i["page"],
                        "width": i["width"], "height": i["height"]}
                       for i in doc.images],
        }
    else:
        item["document"] = None
    item["strategies"] = STRATEGIES
    if item["can_manage"]:
        item["grants"] = db.list_grants(kb_id)
    return item


class RetrievalCfg(BaseModel):
    mode: str | None = None              # semantic | fulltext | hybrid
    top_k: int | None = None
    score_threshold: float | None = None
    bm25_weight: float | None = None
    query_rewrite: bool | None = None
    rerank: bool | None = None
    route_threshold: float | None = None   # 意图识别:语义路由命中阈值(默认 0.45)
    oos_words: list[str] | None = None     # 超范围词表(覆盖内置默认;空=恢复默认)
    image_caption: bool | None = None      # 图片描述(视觉模型,默认开;关闭后图片只留位置标记)


class KBPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    visibility: str | None = None
    retrieval: RetrievalCfg | None = None


@app.patch("/api/kbs/{kb_id}")
def update_kb(kb_id: int, body: KBPatch, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    if "retrieval" in patch:
        cur = (db.get_kb(kb_id) or {}).get("retrieval", {})
        rc = {**cur, **{k: v for k, v in patch.pop("retrieval").items() if v is not None}}
        # mode 仅在本库已配置且本次值非法时拒绝;从未配置过(老库)保持缺省,
        # 检索节点按 semantic 默认处理——否则部分字段更新(如只调阈值)会被 400
        if "mode" in rc and rc["mode"] not in ("semantic", "fulltext", "hybrid"):
            raise HTTPException(400, "检索模式仅支持 semantic / fulltext / hybrid")
        rc["top_k"] = max(1, min(20, int(rc.get("top_k", 6))))
        rc["bm25_weight"] = max(0.0, min(1.0, float(rc.get("bm25_weight", 0.4))))
        rc["score_threshold"] = max(0.0, min(1.0, float(rc.get("score_threshold", 0.3))))
        if "route_threshold" in rc and rc["route_threshold"] is not None:
            rc["route_threshold"] = max(0.0, min(1.0, float(rc["route_threshold"])))
        if "oos_words" in rc:
            # 词表净化:去空白/去重/限量;空列表 = 删除自定义,恢复内置默认词表
            words = [w.strip()[:20] for w in (rc["oos_words"] or []) if str(w).strip()]
            if words:
                rc["oos_words"] = list(dict.fromkeys(words))[:50]
            else:
                rc.pop("oos_words", None)
        patch["retrieval"] = rc
    db.update_kb(kb_id, patch)
    return {"kb": db.get_kb(kb_id)}


@app.delete("/api/kbs/{kb_id}")
def delete_kb(kb_id: int, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    if kb_id == 1:
        raise HTTPException(400, "内置知识库不可删除")
    VectorStore.get().drop_kb(kb_id)
    kdir = config.kb_dir(kb_id)
    shutil.rmtree(kdir, ignore_errors=True)   # 含 images/ 子目录,递归删除
    db.delete_kb(kb_id)
    db.audit_log(user["id"], user["username"], "kb_delete", f"kb={kb_id}")
    return {"ok": True}


@app.post("/api/kbs/{kb_id}/upload")
def upload_document(kb_id: int, file: UploadFile = File(...),
                    user: dict = Depends(current_user)):
    """上传文档(仅 PDF/DOCX/XLSX 文本类,路线二前置:剔除图片OCR/CAD)。
    大小与扩展名双重校验;解析成功后登记 documents 台账(双库一致性锚点)。"""
    kb = _manage_kb(user, kb_id)
    name = Path(file.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in config.UPLOAD_EXTS:
        raise HTTPException(400, f"仅支持 {'/'.join(sorted(config.UPLOAD_EXTS))} 文件")
    limit = config.UPLOAD_MAX_MB * 1024 * 1024
    data = file.file.read(limit + 1)     # 限量读:超限即拒,不全量进内存(大文件 DoS)
    if len(data) > limit:
        raise HTTPException(400, f"文件超过大小上限({config.UPLOAD_MAX_MB}MB)")
    if not data:
        raise HTTPException(400, "文件为空")
    kdir = config.kb_dir(kb_id)
    dest = config.UPLOAD_DIR / f"kb{kb_id}_{int(time.time())}_{name}"
    dest.write_bytes(data)
    # 换文档即换全部图片:img_id 按位置命名,旧图/旧描述对新文档是张冠李戴,
    # 一律清空重来(解析前清,避免新图被误删)
    shutil.rmtree(kdir / "images", ignore_errors=True)
    (kdir / "image_captions.json").unlink(missing_ok=True)
    try:
        parsed = parse_document(dest, Path(name).stem,
                                images_dir=kdir / "images")
    except Exception as e:
        raise HTTPException(400, f"文档解析失败: {e}")
    if not parsed.full_text.strip() and not parsed.tables:
        raise HTTPException(400, "文档未解析出文本、表格或图片内容")
    save_parsed(parsed, kdir / "document.json")
    try:
        for old in kdir.glob("source.*"):   # 清掉旧格式源文件,避免遗留旧文档
            old.unlink(missing_ok=True)
        (kdir / f"source{ext}").write_bytes(data)
    except Exception:
        pass
    # 文档变更:索引与切分缓存失效
    VectorStore.get().clear(kb_id)
    cc = _chunk_cache_path(kb_id)
    if cc.exists():
        cc.unlink()
    db.update_kb(kb_id, {"doc_name": parsed.name, "chunk_count": 0, "built_at": 0})
    doc_id = db.add_document(kb_id, parsed.name, parsed.page_count,
                             len(parsed.full_text), len(parsed.tables), user["username"])
    db.audit_log(user["id"], user["username"], "upload",
                 f"kb={kb_id} file={name} pages={parsed.page_count}")
    analytics.record("upload", question=parsed.name, kb_id=kb_id, kb_name=kb["name"],
                     user_id=user["id"], username=user["username"])
    return {"ok": True, "name": parsed.name, "page_count": parsed.page_count,
            "table_count": len(parsed.tables), "image_count": len(parsed.images),
            "doc_id": doc_id}


# ------------------------------------------------------------------ 切分预览

class PreviewIn(BaseModel):
    strategy: str
    params: dict = {}
    offset: int = 0
    limit: int = 60


@app.post("/api/kbs/{kb_id}/chunks/preview")
def preview_chunks(kb_id: int, body: PreviewIn, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    doc = _require_doc(kb_id)
    # 预览只应用已缓存的图片描述,不触发视觉模型调用(生成是构建路径的事)
    doc_c, cap_sig, note = tasks.doc_for_chunking(kb_id, doc, generate=False)
    key = _cache_key(body.strategy, body.params, cap_sig)
    chunks = _chunk_cache_load(kb_id).get(key)
    if chunks is None:
        t0 = time.time()
        try:
            embedder = Embedder.get() if body.strategy == "semantic" else None
            chunks = run_strategy(body.strategy, doc_c, embedder=embedder, params=body.params)
        except ValueError as e:
            raise HTTPException(400, str(e))
        _chunk_cache_save(kb_id, key, chunks)
        elapsed = round(time.time() - t0, 1)
    else:
        elapsed = 0.0
    stats = chunk_stats(chunks)
    stats["elapsed_s"] = elapsed
    if note:
        stats["note"] = note
    return {
        "stats": stats,
        "total": len(chunks),
        "chunks": [c.to_dict(i) for i, c in
                   enumerate(chunks[body.offset:body.offset + body.limit], start=body.offset)],
    }


# ------------------------------------------------------------------ 索引构建

class BuildIn(BaseModel):
    strategy: str = "structural"
    params: dict = {}


@app.post("/api/kbs/{kb_id}/index/build")
def build_index(kb_id: int, body: BuildIn, user: dict = Depends(current_user)):
    """提交构建任务:standalone 走后台线程,cluster 走 Celery(独立 worker)。
    进度统一经 cache.py 存取,前端轮询 /index/progress——多实例下任意实例可查。"""
    _manage_kb(user, kb_id)
    docs = db.list_documents(kb_id)
    doc_id = str(docs[-1]["id"]) if docs else ""   # 单文档库:与最新台账记录对齐
    try:
        mode = tasks.submit_build(kb_id, body.strategy, body.params,
                                  user["id"], user["username"], doc_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    db.audit_log(user["id"], user["username"], "build_submit",
                 f"kb={kb_id} strategy={body.strategy} mode={mode}")
    return {"ok": True, "mode": mode}


@app.get("/api/kbs/{kb_id}/index/progress")
def index_progress(kb_id: int, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    return tasks.build_progress(kb_id)


@app.get("/api/kbs/{kb_id}/index/status")
def index_status(kb_id: int, user: dict = Depends(current_user)):
    _access_kb(user, kb_id)
    return VectorStore.get().status(kb_id)


@app.delete("/api/kbs/{kb_id}/index")
def clear_index(kb_id: int, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    VectorStore.get().clear(kb_id)
    db.update_kb(kb_id, {"chunk_count": 0, "built_at": 0, "strategy": ""})
    return {"ok": True}


# ------------------------------------------------------------------ 授权管理

@app.get("/api/kbs/{kb_id}/grants")
def kb_grants(kb_id: int, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    return {"grants": db.list_grants(kb_id)}


class GrantIn(BaseModel):
    target_type: str        # user | department
    target_value: str


@app.post("/api/kbs/{kb_id}/grants")
def add_grant(kb_id: int, body: GrantIn, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    if not body.target_value.strip():
        raise HTTPException(400, "授权对象不能为空")
    try:
        grant = db.add_grant(kb_id, body.target_type, body.target_value.strip())
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.audit_log(user["id"], user["username"], "grant",
                 f"kb={kb_id} {body.target_type}:{body.target_value.strip()}")
    return {"grant": grant}


@app.delete("/api/kbs/{kb_id}/grants/{grant_id}")
def remove_grant(kb_id: int, grant_id: int, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    db.remove_grant(grant_id)
    db.audit_log(user["id"], user["username"], "grant_revoke",
                 f"kb={kb_id} grant_id={grant_id}")
    return {"ok": True}


# ------------------------------------------------------------------ 检索/问答

class RetrieveIn(BaseModel):
    query: str
    top_k: int = 6


@app.post("/api/kbs/{kb_id}/retrieve")
def retrieve_only(kb_id: int, body: RetrieveIn, user: dict = Depends(current_user)):
    _access_kb(user, kb_id)
    kb = db.get_kb(kb_id) or {}
    rc = {**(kb.get("retrieval") or {}), "top_k": body.top_k}
    from .retrieval import hybrid_search
    vec = Embedder.get().embed_query(body.query) if rc.get("mode") != "fulltext" else None
    hits, debug = hybrid_search(kb_id, body.query, vec, rc, VectorStore.get())
    return {"hits": hits, "debug": debug}


@app.get("/api/kbs/{kb_id}/images/{img_id}")
def kb_image(kb_id: int, img_id: str, user: dict = Depends(current_user)):
    """知识库文档抽取的原图(用于问答引用溯源)。可读该库的用户均可查看。"""
    _access_kb(user, kb_id)
    if not re.fullmatch(r"[\w-]{1,64}", img_id):
        raise HTTPException(400, "非法图片标识")
    doc = load_parsed(config.kb_dir(kb_id) / "document.json")
    infos = {i["img_id"]: i for i in (doc.images if doc else [])}
    info = infos.get(img_id)
    if not info:
        raise HTTPException(404, "图片不存在")
    path = config.kb_dir(kb_id) / info["file"]
    if not path.exists():
        raise HTTPException(404, "图片文件缺失")
    return FileResponse(path)


# ---------------------------------------------------------------- 意图路由

class IntentIn(BaseModel):
    name: str
    label: str
    utterances: list[str] = []
    boost_keywords: list[str] = []


class IntentPatch(BaseModel):
    label: str | None = None
    utterances: list[str] | None = None
    boost_keywords: list[str] | None = None


@app.get("/api/kbs/{kb_id}/intents")
def list_intents(kb_id: int, user: dict = Depends(current_user)):
    _access_kb(user, kb_id)
    return {"intents": db.list_intents(kb_id)}


@app.post("/api/kbs/{kb_id}/intents")
def create_intent(kb_id: int, body: IntentIn, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    try:
        it = db.add_intent(kb_id, body.name, body.label, body.utterances, body.boost_keywords)
    except ValueError as e:
        raise HTTPException(400, str(e))
    SemanticRouter.invalidate(kb_id)
    return {"intent": it}


@app.patch("/api/kbs/{kb_id}/intents/{intent_id}")
def patch_intent(kb_id: int, intent_id: int, body: IntentPatch,
                 user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    try:
        db.update_intent(intent_id, {k: v for k, v in body.model_dump().items() if v is not None})
    except ValueError as e:
        raise HTTPException(400, str(e))
    SemanticRouter.invalidate(kb_id)
    return {"ok": True}


@app.delete("/api/kbs/{kb_id}/intents/{intent_id}")
def delete_intent(kb_id: int, intent_id: int, user: dict = Depends(current_user)):
    _manage_kb(user, kb_id)
    db.remove_intent(intent_id)
    SemanticRouter.invalidate(kb_id)
    return {"ok": True}


class ChatIn(BaseModel):
    message: str
    kb_id: int = 1
    session_id: str | None = None
    stream: bool = True


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@app.post("/api/chat")
def chat(body: ChatIn, user: dict = Depends(current_user)):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "消息不能为空")
    kb = _access_kb(user, body.kb_id)
    settings = config.load_settings()
    if not settings["deepseek_api_key"] and not settings.get("deepseek_backup_api_key"):
        raise HTTPException(400, "请先在「设置」页配置 DeepSeek API Key")
    status = VectorStore.get().status(body.kb_id)
    if not status.get("indexed"):
        raise HTTPException(400, "该知识库尚未构建索引,请联系管理员在「知识库」页构建")

    agent = RAGAgent.get()
    session_id = body.session_id or uuid.uuid4().hex[:12]
    thread_id = thread_id_of(user["id"], body.kb_id, session_id)
    # user_id 入 configurable:kb_stats 工具按用户可见范围统计,不越权
    cfg = {"configurable": {"thread_id": thread_id, "kb_id": body.kb_id,
                            "user_id": user["id"]}}
    from langchain_core.messages import HumanMessage
    inputs = {"messages": [HumanMessage(content=message)]}
    t0 = time.time()

    def _record(answer: str, usage: dict, sources: list, intent: str, latency: int,
                route_name: str = "", key_route: str = "") -> int:
        analytics.record("chat", session_id=session_id, question=message,
                         answer_preview=answer, usage=usage, latency_ms=latency,
                         sources_n=len(sources), strategy=status.get("strategy", ""),
                         user_id=user["id"], username=user["username"],
                         kb_id=body.kb_id, kb_name=kb["name"],
                         # 意图分布统计用原始路由名(如 discipline),无则退化为分支名
                         intent=route_name or intent,
                         # 主/备 LLM 通道标记,运维据此观测备用 Key 触发频次
                         key_route=key_route)
        # 刷新会话列表元数据(独立小表,避免列表接口扫描全量 checkpoints);
        # turns 由表内自增维护,返回最新轮数
        try:
            return db.upsert_chat_session(user["id"], body.kb_id, session_id,
                                          title=message)
        except Exception:
            return 0

    if not body.stream:
        result = agent.graph.invoke(inputs, cfg)
        last_msg = result["messages"][-1]
        answer = str(last_msg.content)
        # 推理模型思考过程(模板拒答等无 LLM 分支为空串)
        thinking = str(getattr(last_msg, "additional_kwargs", {}).get("reasoning_content") or "")
        usage = result.get("usage", {})
        intent = result.get("intent", "kb_query")
        latency = int((time.time() - t0) * 1000)
        sources = result.get("sources", []) if intent == "kb_query" else []
        turns = _record(answer, usage, sources, intent, latency,
                        result.get("route_name", ""), result.get("key_route", ""))
        return {
            "session_id": session_id, "answer": answer, "thinking": thinking,
            "sources": sources, "usage": usage,
            "intent": intent,
            "intent_label": result.get("intent_label") or INTENT_LABELS.get(intent, intent),
            "intent_score": result.get("intent_score", 0.0),
            "intent_source": result.get("intent_source", ""),
            "route_name": result.get("route_name", ""),
            "latency_ms": latency, "memory_turns": turns,
            "key_route": result.get("key_route", ""),
            "retrieval": result.get("retrieval_debug", {}),
            "low_relevance": bool(result.get("low_relevance")),
            "rewritten_query": result.get("rewritten_query", ""),
            "tool_calls": result.get("tool_calls", []),
        }

    def gen():
        answer_buf = []
        thinking_buf = []
        final_values: dict = {}
        try:
            for mode, payload in agent.graph.stream(
                    inputs, cfg, stream_mode=["messages", "values", "custom"]):
                if mode == "messages":
                    chunk, meta = payload
                    if meta.get("langgraph_node") in (
                            "generate", "generate_chitchat", "tool_agent"):
                        # 推理模型(deepseek-reasoner 等)的思考过程先行下发,
                        # 与正式回答分通道传输,前端独立折叠展示
                        rc = (chunk.additional_kwargs or {}).get("reasoning_content")
                        if rc:
                            thinking_buf.append(rc)
                            yield _sse({"type": "thinking", "content": rc})
                        if chunk.content:
                            answer_buf.append(chunk.content)
                            yield _sse({"type": "token", "content": chunk.content})
                elif mode == "custom":
                    # 工具节点的实时进度事件(tool_agent 内 get_stream_writer 发出)
                    if isinstance(payload, dict) and payload.get("type"):
                        yield _sse(payload)
                elif mode == "values":
                    final_values = payload
            usage = final_values.get("usage", {})
            intent = final_values.get("intent", "kb_query")
            sources = final_values.get("sources", []) if intent == "kb_query" else []
            latency = int((time.time() - t0) * 1000)
            answer = "".join(answer_buf)
            # 拒答等无流式输出的分支:取最终消息文本
            if not answer and final_values.get("messages"):
                answer = str(final_values["messages"][-1].content)
            turns = _record(answer, usage, sources, intent, latency,
                            final_values.get("route_name", ""),
                            final_values.get("key_route", ""))
            yield _sse({
                "type": "done", "session_id": session_id, "answer": answer,
                "thinking": "".join(thinking_buf),
                "sources": sources, "usage": usage,
                "intent": intent,
                "intent_label": final_values.get("intent_label") or INTENT_LABELS.get(intent, intent),
                "intent_score": final_values.get("intent_score", 0.0),
                "intent_source": final_values.get("intent_source", ""),
                "route_name": final_values.get("route_name", ""),
                "latency_ms": latency, "memory_turns": turns,
                "key_route": final_values.get("key_route", ""),
                "retrieval": final_values.get("retrieval_debug", {}),
                "low_relevance": bool(final_values.get("low_relevance")),
                "rewritten_query": final_values.get("rewritten_query", ""),
                "tool_calls": final_values.get("tool_calls", []),
            })
        except Exception as e:
            yield _sse({"type": "error", "error": str(e)[:300]})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/chat/sessions")
def chat_sessions(kb_id: int = 1, user: dict = Depends(current_user)):
    _access_kb(user, kb_id)
    return {"sessions": RAGAgent.get().list_sessions(user["id"], kb_id)}


@app.get("/api/chat/history/{kb_id}/{session_id}")
def chat_history(kb_id: int, session_id: str, user: dict = Depends(current_user)):
    _access_kb(user, kb_id)
    tid = thread_id_of(user["id"], kb_id, session_id)
    return {"messages": RAGAgent.get().history(tid),
            # 轮数读会话元数据表,不为统计去反序列化整条 checkpoint 状态
            "memory_turns": db.get_chat_session_turns(user["id"], kb_id, session_id)}


@app.delete("/api/chat/sessions/{kb_id}/{session_id}")
def chat_delete(kb_id: int, session_id: str, user: dict = Depends(current_user)):
    tid = thread_id_of(user["id"], kb_id, session_id)
    RAGAgent.get().delete_session(tid)
    return {"ok": True}


# ------------------------------------------------------------------ 统计(管理员)

@app.get("/api/stats/summary")
def stats_summary(_: dict = Depends(require_manager)):
    return analytics.summary()


@app.get("/api/stats/timeseries")
def stats_timeseries(_: dict = Depends(require_manager)):
    return {"days": analytics.timeseries(14)}


def _intent_labels() -> dict:
    """原始路由名 → 展示标签(内置三类 + 各知识库自定义意图)。"""
    m = {i["name"]: i["label"] for kb in db.list_kbs() for i in db.list_intents(kb["id"])}
    m.update(INTENT_LABELS)
    return m


@app.get("/api/stats/recent")
def stats_recent(_: dict = Depends(require_manager)):
    items = analytics.recent(15)
    m = _intent_labels()
    for it in items:
        it["intent_label"] = m.get(it.get("intent"), it.get("intent"))
    return {"items": items}


@app.get("/api/stats/intents")
def stats_intents(_: dict = Depends(require_manager)):
    items = analytics.by_intent()
    # 埋点记录的是原始路由名(如 discipline),补上映射的展示标签
    m = _intent_labels()
    for it in items:
        it["label"] = m.get(it["intent"], it["intent"])
    return {"items": items}


@app.get("/api/stats/kbs")
def stats_kbs(_: dict = Depends(require_manager)):
    return {"items": analytics.by_kb()}


# ------------------------------------------------------------------ 设置(系统管理员)

class SettingsIn(BaseModel):
    deepseek_api_key: str | None = None
    deepseek_backup_api_key: str | None = None
    deepseek_base_url: str | None = None
    deepseek_model: str | None = None
    vision_model: str | None = None      # 图片描述用视觉模型(须支持图片输入)
    retrieve_top_k: int | None = None
    tools_enabled: bool | None = None    # 工具调用总开关
    tools_model: str | None = None       # 工具链路模型(须支持 function calling)
    mcp_servers: str | None = None       # MCP servers JSON 字符串(空=仅内置工具)
    skills_enabled: bool | None = None   # 技能(Skills)总开关


@app.get("/api/settings")
def get_settings(_: dict = Depends(require_admin)):
    s = config.load_settings()
    return {
        "deepseek_api_key_masked": config.mask_key(s["deepseek_api_key"]),
        "deepseek_api_key_set": bool(s["deepseek_api_key"]),
        "deepseek_backup_api_key_masked": config.mask_key(
            s.get("deepseek_backup_api_key", "")),
        "deepseek_backup_api_key_set": bool(s.get("deepseek_backup_api_key")),
        "deepseek_base_url": s["deepseek_base_url"],
        "deepseek_model": s["deepseek_model"],
        "vision_model": s.get("vision_model", "deepseek-flash"),
        "embedding_model": s["embedding_model"],
        "memory_turns": s["memory_turns"],
        "retrieve_top_k": s["retrieve_top_k"],
        "tools_enabled": bool(s.get("tools_enabled", True)),
        "tools_model": s.get("tools_model", "deepseek-chat"),
        "mcp_servers": s.get("mcp_servers", ""),
        "skills_enabled": bool(s.get("skills_enabled", True)),
    }


@app.post("/api/settings")
def update_settings(body: SettingsIn, _: dict = Depends(require_admin)):
    patch = {k: v for k, v in body.model_dump().items() if v is not None and v != ""}
    if "****" in patch.get("deepseek_api_key", ""):
        patch.pop("deepseek_api_key")  # 忽略掩码回填
    if "****" in patch.get("deepseek_backup_api_key", ""):
        patch.pop("deepseek_backup_api_key")
    # 布尔字段会被上面的 v != "" 过滤误杀(False 是有意义的值),补回
    if body.tools_enabled is not None:
        patch["tools_enabled"] = body.tools_enabled
    if body.skills_enabled is not None:
        patch["skills_enabled"] = body.skills_enabled
    if body.mcp_servers is not None:
        # 允许空串清空;非空必须是合法 JSON 对象
        raw = (body.mcp_servers or "").strip()
        if raw:
            try:
                if not isinstance(json.loads(raw), dict):
                    raise ValueError
            except ValueError:
                raise HTTPException(400, "MCP servers 必须是 JSON 对象,如 "
                                    '{"name": {"url": "http://.../sse"}}')
        patch["mcp_servers"] = raw
    s = config.save_settings(patch)
    try:
        RAGAgent.reload()   # 设置进了图构建闭包,必须重建才生效;失败不阻塞保存
        SemanticRouter.invalidate()   # tools_enabled 影响路由加载,缓存同步失效
    except Exception:
        logging.getLogger(__name__).warning("设置已保存,但重建 RAG 图失败", exc_info=True)
    return {"ok": True, "deepseek_api_key_set": bool(s["deepseek_api_key"])}


@app.post("/api/settings/test")
def test_settings(_: dict = Depends(require_admin)):
    s = config.load_settings()
    test_key = s["deepseek_api_key"] or s.get("deepseek_backup_api_key", "")
    if not test_key:
        return {"ok": False, "error": "尚未配置 API Key"}
    t0 = time.time()
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=s["deepseek_model"], base_url=s["deepseek_base_url"],
                         api_key=test_key, max_tokens=8, timeout=30)
        resp = llm.invoke("你好,请回复“连接正常”")
        return {"ok": True, "latency_ms": int((time.time() - t0) * 1000),
                "reply": str(resp.content)[:50], "model": s["deepseek_model"]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300], "latency_ms": int((time.time() - t0) * 1000)}


# ------------------------------------------------------------------ 技能(Skills,系统管理员)


@app.get("/api/skills")
def list_skills(_: dict = Depends(require_admin)):
    return {"items": skills_mod.list_skills()}


@app.post("/api/skills")
async def upload_skill(file: UploadFile = File(...), _: dict = Depends(require_admin)):
    data = await file.read()
    try:
        return skills_mod.save_upload(file.filename or "", data)
    except ValueError as e:
        raise HTTPException(400, str(e))


class SkillPatch(BaseModel):
    enabled: bool


@app.patch("/api/skills/{name}")
def patch_skill(name: str, body: SkillPatch, _: dict = Depends(require_admin)):
    if not skills_mod.set_enabled(name, body.enabled):
        raise HTTPException(404, "技能不存在")
    return {"ok": True}


@app.delete("/api/skills/{name}")
def delete_skill(name: str, _: dict = Depends(require_admin)):
    try:
        skills_mod.delete_skill(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


# ------------------------------------------------------------------ 业务系统数据对接(1.5 工具管理)


class ConnectorIn(BaseModel):
    name: str
    label: str = ""
    description: str = ""
    type: str = "rest"                # rest | database
    config: dict = {}
    enabled: bool = True


class ConnectorPatch(BaseModel):
    name: str | None = None
    label: str | None = None
    description: str | None = None
    config: dict | None = None
    enabled: bool | None = None


class ConnectorPreviewIn(BaseModel):
    type: str = "rest"
    config: dict = {}
    args: dict = {}


@app.get("/api/connectors")
def list_connectors(_: dict = Depends(require_manager)):
    return {"items": db.list_connectors()}


@app.post("/api/connectors")
def create_connector(body: ConnectorIn, user: dict = Depends(require_manager)):
    try:
        cfg = connectors_mod.validate_config(body.type, body.config)
    except connectors_mod.ConnectorError as e:
        raise HTTPException(400, str(e))
    try:
        conn = db.create_connector(body.name, body.label, body.description,
                                   body.type, cfg, body.enabled, user["username"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    if conn is None:
        raise HTTPException(409, "连接器名称已存在")
    db.audit_log(user["id"], user["username"], "connector_create",
                 f"{conn['name']}({conn['type']})")
    return conn


@app.patch("/api/connectors/{cid}")
def update_connector(cid: int, body: ConnectorPatch,
                     user: dict = Depends(require_manager)):
    conn = db.get_connector(cid)
    if not conn:
        raise HTTPException(404, "连接器不存在")
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    if "config" in patch:
        try:
            patch["config"] = connectors_mod.validate_config(
                conn["type"], patch["config"])
        except connectors_mod.ConnectorError as e:
            raise HTTPException(400, str(e))
    try:
        db.update_connector(cid, patch)
    except ValueError as e:
        raise HTTPException(400, str(e))
    db.audit_log(user["id"], user["username"], "connector_update",
                 f"{conn['name']}: {sorted(patch.keys())}")
    return db.get_connector(cid)


@app.delete("/api/connectors/{cid}")
def delete_connector(cid: int, user: dict = Depends(require_manager)):
    conn = db.get_connector(cid)
    if not conn:
        raise HTTPException(404, "连接器不存在")
    db.delete_connector(cid)
    db.audit_log(user["id"], user["username"], "connector_delete", conn["name"])
    return {"ok": True}


class ConnectorTestIn(BaseModel):
    args: dict = {}


@app.post("/api/connectors/{cid}/test")
def test_connector(cid: int, body: ConnectorTestIn,
                   user: dict = Depends(require_manager)):
    """已保存连接器的「测试连接」:返回截断后的数据预览(行/列/单元格均有限)。"""
    conn = db.get_connector(cid)
    if not conn:
        raise HTTPException(404, "连接器不存在")
    result = connectors_mod.preview_connector(conn["type"], conn["config"], body.args)
    db.audit_log(user["id"], user["username"], "connector_test",
                 f"{conn['name']}: {'ok' if result['ok'] else result.get('error', '')[:200]}")
    return result


@app.post("/api/connectors/preview")
def preview_unsaved_connector(body: ConnectorPreviewIn,
                              user: dict = Depends(require_manager)):
    """表单内「测试连接」(未保存的配置):用户点击即可预览接口/查询数据。"""
    result = connectors_mod.preview_connector(body.type, body.config, body.args)
    db.audit_log(user["id"], user["username"], "connector_preview",
                 f"{body.type}: {'ok' if result['ok'] else result.get('error', '')[:200]}")
    return result


# ------------------------------------------------------------------ 健康

@app.get("/api/health")
def health():
    s = config.load_settings()
    return {
        "ok": True,
        "llm_configured": bool(s["deepseek_api_key"]),
        "llm_model": s["deepseek_model"],
        "embedding_ready": Embedder.get().ready,
        "kbs": len(db.list_kbs()),
    }


# ------------------------------------------------------------------ 前端托管(SPA)

@app.on_event("startup")
def _startup():
    db.init_db()
    threading.Thread(target=ensure_kb_document, args=(1,), daemon=True).start()


@app.exception_handler(HTTPException)
async def http_exc(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


if config.FRONTEND_DIST.exists():
    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("api"):
            raise HTTPException(404, "Not Found")
        target = (config.FRONTEND_DIST / full_path).resolve()
        if target.is_file() and str(target).startswith(str(config.FRONTEND_DIST)):
            return FileResponse(target)
        return FileResponse(config.FRONTEND_DIST / "index.html")
