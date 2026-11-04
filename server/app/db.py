"""账号与权限数据层:用户 / 会话令牌 / 知识库 / 授权(对应规格书 1.7 权限管理)。

- RBAC 三种内置角色:admin(系统管理员)/ workspace_admin(工作空间管理员)/ user(普通用户)。
- 资源授权:知识库可按「用户」或「部门」两个维度授权,支持公开/私有可见性。
- 访问判定优先级(高 → 低):用户级授权 > 部门级授权 > 公开可见;管理员角色永远放行。
- 口令使用 PBKDF2-HMAC-SHA256(10 万次迭代)加盐哈希存储;登录令牌为随机 UUID,7 天过期。
- LDAP 认证为预留扩展点(见 authenticate() 注释)。

存储层(路线二改造):SQLAlchemy Core,双方言一套代码——
- standalone:SQLite(零依赖,默认);
- cluster:IvorySQL(PG 协议完全兼容,DB_URI=postgresql+psycopg://... 注入)。
  Integer 自增主键在 PG 方言自动编译为 SERIAL,SQLite 下为 AUTOINCREMENT;
  幂等写入用方言 ON CONFLICT,无方言 SQL 硬编码。
- 时间戳统一用 epoch 秒(Float),规避双方言时区类型差异;生产如需
  TIMESTAMPTZ 可按列 ALTER,代码无需改动。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid

from sqlalchemy import (Column, Float, Integer, MetaData, String, Table, Text,
                        UniqueConstraint, create_engine, delete, func, insert, select,
                        text, update)
from sqlalchemy.exc import IntegrityError

from . import config

logger = logging.getLogger(__name__)

ROLES = {
    "admin": "系统管理员",
    "workspace_admin": "工作空间管理员",
    "user": "普通用户",
}

DEPARTMENTS = [
    "信息化办公室", "综合管理部", "生产制造部", "工艺技术部",
    "质量部", "设备动力部", "财务部", "人力资源部",
]

_lock = threading.Lock()   # 仅用于种子初始化等低频临界区

# ---------------------------------------------------------------- 表定义

_meta = MetaData()

users = Table(
    "users", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),   # PG 方言 → SERIAL
    Column("username", String(64), unique=True, nullable=False),
    Column("password_hash", String(128), nullable=False),
    Column("display_name", String(64), default=""),
    Column("department", String(64), default=""),
    Column("role", String(32), nullable=False, default="user"),
    Column("status", Integer, nullable=False, default=1),
    Column("created_at", Float, nullable=False),
)

auth_sessions = Table(        # 登录令牌(物理表名 sessions,与旧库一致)
    "sessions", _meta,
    Column("token", String(64), primary_key=True),
    Column("user_id", Integer, nullable=False, index=True),
    Column("expires_at", Float, nullable=False),
)

knowledge_bases = Table(
    "knowledge_bases", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(128), nullable=False),
    Column("description", Text, default=""),
    Column("doc_name", String(256), default=""),
    Column("visibility", String(16), nullable=False, default="private"),
    Column("owner_id", Integer, nullable=False),
    Column("strategy", String(64), default=""),
    Column("chunk_count", Integer, default=0),
    Column("built_at", Float, default=0),
    Column("retrieval_config", Text, default=""),   # JSON:混合检索/阈值/精排等
    Column("created_at", Float, nullable=False),
)

kb_grants = Table(
    "kb_grants", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kb_id", Integer, nullable=False, index=True),
    Column("target_type", String(16), nullable=False),      # user | department
    Column("target_value", String(128), nullable=False),
    Column("created_at", Float, nullable=False),
    UniqueConstraint("kb_id", "target_type", "target_value", name="uq_grant"),
)

kb_intents = Table(
    "kb_intents", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kb_id", Integer, nullable=False, index=True),   # 0 = 全局内置路由
    Column("name", String(64), nullable=False),
    Column("label", String(64), nullable=False),
    Column("utterances", Text, default="[]"),               # JSON:示例句
    Column("boost_keywords", Text, default="[]"),           # JSON:检索增强关键词
    Column("created_at", Float, nullable=False),
    UniqueConstraint("kb_id", "name", name="uq_intent"),
)

chat_sessions = Table(        # 会话列表元数据(不扫 checkpoint 大表)
    "chat_sessions", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=False),
    Column("kb_id", Integer, nullable=False),
    Column("session_id", String(64), nullable=False),
    Column("title", String(128), default=""),
    Column("turns", Integer, default=0),
    Column("updated_at", Float, nullable=False, index=True),
    UniqueConstraint("user_id", "kb_id", "session_id", name="uq_chat_session"),
)

documents = Table(            # 文档元数据(路线二:文档级删除/审计的锚点)
    "documents", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kb_id", Integer, nullable=False, index=True),
    Column("filename", String(256), nullable=False),
    Column("pages", Integer, default=0),
    Column("chars", Integer, default=0),
    Column("tables_count", Integer, default=0),
    Column("uploaded_by", String(64), default=""),
    Column("created_at", Float, nullable=False),
)

audit_logs = Table(           # 审计:敏感词拦截/授权变更等(路线二 七.4)
    "audit_logs", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, nullable=False, index=True),
    Column("user_id", Integer, default=0),
    Column("username", String(64), default=""),
    Column("action", String(64), nullable=False),           # sensitive_block / grant / ...
    Column("detail", Text, default=""),
)

events = Table(               # 埋点统计(问答/构建/上传),analytics.py 使用
    "events", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, nullable=False, index=True),
    Column("kind", String(16), nullable=False, index=True),  # chat | build | upload
    Column("session_id", String(64), default=""),
    Column("question", Text, default=""),
    Column("answer_preview", Text, default=""),
    Column("prompt_tokens", Integer, default=0),
    Column("completion_tokens", Integer, default=0),
    Column("total_tokens", Integer, default=0),
    Column("latency_ms", Integer, default=0),
    Column("sources_n", Integer, default=0),
    Column("strategy", String(64), default=""),
    Column("user_id", Integer, default=0),
    Column("username", String(64), default=""),
    Column("kb_id", Integer, default=0),
    Column("kb_name", String(128), default=""),
    Column("intent", String(64), default=""),
    Column("key_route", String(16), default=""),   # 本次回答的 LLM 通道(主/备 Key)
)

connectors = Table(           # 业务系统数据对接(对应规格书 1.5 工具管理/1.6 业务智能体)
    "connectors", _meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(64), unique=True, nullable=False),  # 英文标识,工具名 biz_<name>
    Column("label", String(64), default=""),                  # 展示名(界面/审计用)
    Column("description", Text, default=""),                  # 供 LLM 判断何时调用
    Column("type", String(16), nullable=False, default="rest"),  # rest | database
    Column("config", Text, default="{}"),                     # JSON 连接配置(见 connectors.py)
    Column("enabled", Integer, nullable=False, default=1),    # 1 启用 0 停用(工具启停)
    Column("created_by", String(64), default=""),
    Column("created_at", Float, nullable=False),
)


# ---------------------------------------------------------------- 引擎

_engine = None
_engine_lock = threading.Lock()


def engine():
    """惰性创建引擎。cluster(IvorySQL)配连接池,防止多实例把数据库连接打爆。"""
    global _engine
    with _engine_lock:
        if _engine is None:
            uri = config.DB_URI
            if uri.startswith("sqlite"):
                _engine = create_engine(
                    uri, connect_args={"check_same_thread": False}, pool_pre_ping=True)
            else:
                _engine = create_engine(
                    uri, pool_size=10, max_overflow=20, pool_pre_ping=True,
                    pool_recycle=1800,
                )
            logger.info("DB 引擎就绪: %s", uri.split("://")[0])
        return _engine


def _is_pg() -> bool:
    return engine().dialect.name == "postgresql"


def _insert_ignore(table: Table):
    """方言中立的 INSERT ... ON CONFLICT DO NOTHING(替代 SQLite 的 INSERT OR IGNORE)。"""
    if _is_pg():
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        return pg_insert(table).on_conflict_do_nothing()
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    return sqlite_insert(table).on_conflict_do_nothing()


# ---------------------------------------------------------------- 口令

def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    ).hex()
    return f"{salt}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return _hash_password(password, salt) == stored


def _row_user(row) -> dict:
    return {
        "id": row.id, "username": row.username, "display_name": row.display_name,
        "department": row.department, "role": row.role,
        "role_name": ROLES.get(row.role, row.role), "status": row.status,
        "created_at": row.created_at,
    }


# ---------------------------------------------------------------- 初始化/种子数据

# 内置全局路由(kb_id=0,平台管有,与是否开启内置样例数据无关):
# 启动时 upsert——老库自动补齐新增路由(如 tool),内置示例句的修订
# (如数学题从"超出范围"移交"工具调用")也随之生效。 kb 级意图仍在演示种子中。
BUILTIN_ROUTE_SEEDS = [
    ("chitchat", "闲聊",
     ["你好", "您好", "早上好", "谢谢你", "再见", "你是谁", "辛苦了", "在吗"], []),
    ("out_of_scope", "超出范围",
     ["帮我写一段代码", "翻译这句话", "今天天气怎么样", "讲个笑话",
      "最近有什么新闻", "股票行情如何"], []),
    ("tool", "工具调用",
     ["现在几点了", "今天是几号", "帮我计算 128 乘 35", "算一下 1024 除以 8",
      "抓取这个网页的内容", "总结一下这个网址链接",
      "知识库里有多少文档", "统计一下知识库的数量"], []),
]


def _ensure_builtin_routes(conn) -> None:
    """upsert 内置全局路由(kb_id=0)。幂等,启动时调用。"""
    now = time.time()
    for name, label, utt, kw in BUILTIN_ROUTE_SEEDS:
        row = conn.execute(
            select(kb_intents.c.id)
            .where(kb_intents.c.kb_id == 0, kb_intents.c.name == name)).first()
        if row:
            conn.execute(update(kb_intents)
                         .where(kb_intents.c.id == row.id)
                         .values(label=label,
                                 utterances=json.dumps(utt, ensure_ascii=False),
                                 boost_keywords=json.dumps(kw, ensure_ascii=False)))
        else:
            conn.execute(_insert_ignore(kb_intents).values(
                kb_id=0, name=name, label=label,
                utterances=json.dumps(utt, ensure_ascii=False),
                boost_keywords=json.dumps(kw, ensure_ascii=False),
                created_at=now))


def _ensure_columns(eng) -> None:
    """存量库补列(create_all 不 ALTER):SQLite/PG 均支持 ADD COLUMN,幂等。"""
    from sqlalchemy import inspect as sa_inspect
    insp = sa_inspect(eng)
    migrations = {"events": {"key_route": "VARCHAR(16) DEFAULT ''"}}
    with eng.begin() as conn:
        for table, cols in migrations.items():
            if table not in insp.get_table_names():
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for col, decl in cols.items():
                if col not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {decl}"))
                    logger.info("存量库补列: %s.%s", table, col)


def init_db():
    with _lock:
        eng = engine()
        _meta.create_all(eng)
        _ensure_columns(eng)
        with eng.begin() as conn:
            # admin 始终种子(系统入口),初始密码可用 ADMIN_PASSWORD 注入;
            # 内置账号/内置知识库受 SEED_DEMO_DATA 控制——生产置 0,避免弱口令账号外露
            conn.execute(_insert_ignore(users).values(
                username="admin", password_hash=_hash_password(config.ADMIN_PASSWORD),
                display_name="系统管理员", department="信息化办公室", role="admin",
                status=1, created_at=time.time()))
            _ensure_builtin_routes(conn)   # 平台级路由,不随演示数据开关
            if not config.SEED_DEMO_DATA:
                return

            demo_users = [
                ("manager", "manager123", "张主管", "综合管理部",   "workspace_admin"),
                ("chen",    "chen123",    "陈工",   "生产制造部",   "user"),
                ("li",      "li123",      "李工",   "工艺技术部",   "user"),
                ("wang",    "wang123",    "王工",   "设备动力部",   "user"),
            ]
            for username, pwd, display, dept, role in demo_users:
                conn.execute(_insert_ignore(users).values(
                    username=username, password_hash=_hash_password(pwd),
                    display_name=display, department=dept, role=role,
                    status=1, created_at=time.time()))

            admin_id = conn.execute(
                select(users.c.id).where(users.c.username == "admin")).scalar_one()
            manager_id = conn.execute(
                select(users.c.id).where(users.c.username == "manager")).scalar_one()

            # 幂等种子知识库:1=演示文档(公开),2=私有库(演示授权与隔离)
            kb_seeds = [
                (1, "企业知识库",
                 "内置规章制度汇编全文(文本+表格),用于验证多种切分策略与检索质量;正式部署请替换为本单位制度文件",
                 "public", admin_id),
                (2, "内部工作规范", "私有知识库:验证按部门/用户授权与访问隔离;正式部署请替换为内部工艺或技术文件",
                 "private", manager_id),
            ]
            for kb_id, name, desc, vis, owner in kb_seeds:
                exists = conn.execute(
                    select(knowledge_bases.c.id)
                    .where(knowledge_bases.c.id == kb_id)).first()
                if not exists:
                    conn.execute(knowledge_bases.insert().values(
                        id=kb_id, name=name, description=desc, visibility=vis,
                        owner_id=owner, created_at=time.time()))
            if _is_pg():
                # 显式 id 种子不会推进 SERIAL 序列,手动对齐,否则后续自增插 id=1 冲突
                conn.execute(text(
                    "SELECT setval(pg_get_serial_sequence('knowledge_bases','id'),"
                    " COALESCE((SELECT MAX(id) FROM knowledge_bases), 1))"))
            # 种子授权:KB2 授权给「生产制造部」部门 + 用户 li(演示部门级/用户级两条路径)
            for kb_id, ttype, tval in [(2, "department", "生产制造部"), (2, "user", "li")]:
                conn.execute(_insert_ignore(kb_grants).values(
                    kb_id=kb_id, target_type=ttype, target_value=tval,
                    created_at=time.time()))
            # 种子意图路由(semantic-router 模式:示例句为锚点,关键词注入检索)
            # 内置全局路由(kb_id=0)已由 _ensure_builtin_routes 维护,此处仅知识库级
            # 注:示例句为语义路由锚点,与内置样例文档(规章制度汇编)内容对齐;
            # 正式部署替换文档后,应在「智能体编排」页同步调整示例句。
            intent_seeds = [
                (1, "discipline", "违规处理",
                 ["违章操作会受到什么处理", "迟到早退怎么处理", "违反安全规定有什么后果",
                  "上班时间饮酒怎么处罚", "对处理决定不服怎么申诉"],
                 ["处分", "违章", "警告", "记过", "解除劳动合同", "申诉"]),
                (1, "reward", "奖励评定",
                 ["绩效奖金怎么评定", "优秀员工评选条件是什么", "合理化建议奖励标准",
                  "评优评先有什么要求"],
                 ["绩效", "奖金", "优秀员工", "评定", "奖励"]),
                (1, "affair", "事务办理",
                 ["请假手续怎么办", "报销需要哪些材料", "加班申请流程是什么",
                  "出差审批怎么走", "离职手续如何办理"],
                 ["请假", "报销", "加班", "出差", "审批", "离职"]),
                (2, "workflow", "业务流程",
                 ["文件借阅怎么申请", "会议室怎么预定", "账号权限怎么开通",
                  "外来人员来访登记流程"],
                 ["借阅", "预定", "权限", "开通", "登记"]),
            ]
            for kb_id, name, label, utt, kw in intent_seeds:
                conn.execute(_insert_ignore(kb_intents).values(
                    kb_id=kb_id, name=name, label=label,
                    utterances=json.dumps(utt, ensure_ascii=False),
                    boost_keywords=json.dumps(kw, ensure_ascii=False),
                    created_at=time.time()))


# ---------------------------------------------------------------- 认证

def authenticate(username: str, password: str) -> dict | None:
    """本地账号认证。LDAP 集成扩展点:企业部署时可在此处先走 LDAP 绑定认证,
    失败后回退本地口令,或将用户属性(部门/角色)从 LDAP 同步进 users 表。"""
    with engine().connect() as conn:
        row = conn.execute(
            select(users).where(users.c.username == username)).first()
    if not row or not row.status:
        return None
    if not _verify_password(password, row.password_hash):
        return None
    return _row_user(row)


def create_session(user_id: int, days: int = 7) -> str:
    token = uuid.uuid4().hex
    with _lock, engine().begin() as conn:
        conn.execute(delete(auth_sessions).where(auth_sessions.c.expires_at < time.time()))
        conn.execute(auth_sessions.insert().values(
            token=token, user_id=user_id, expires_at=time.time() + days * 86400))
    return token


def user_by_token(token: str) -> dict | None:
    if not token:
        return None
    with engine().connect() as conn:
        row = conn.execute(
            select(users).select_from(
                auth_sessions.join(users, users.c.id == auth_sessions.c.user_id))
            .where(auth_sessions.c.token == token)
            .where(auth_sessions.c.expires_at > time.time())).first()
    if not row or not row.status:
        return None
    return _row_user(row)


def revoke_session(token: str):
    with _lock, engine().begin() as conn:
        conn.execute(delete(auth_sessions).where(auth_sessions.c.token == token))



# ---------------------------------------------------------------- 用户管理

def list_users() -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(select(users).order_by(users.c.id)).all()
    return [_row_user(r) for r in rows]


def get_user(user_id: int) -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(select(users).where(users.c.id == user_id)).first()
    return _row_user(row) if row else None


def create_user(username: str, password: str, display_name: str,
                department: str, role: str) -> dict:
    if role not in ROLES:
        raise ValueError("非法角色")
    if not username:
        raise ValueError("用户名不能为空")
    try:
        with engine().begin() as conn:
            conn.execute(insert(users).values(
                username=username, password_hash=_hash_password(password),
                display_name=display_name or username, department=department,
                role=role, status=1, created_at=time.time()))
    except IntegrityError:
        raise ValueError("用户名已存在")
    with engine().connect() as conn:
        row = conn.execute(select(users).where(users.c.username == username)).first()
    return _row_user(row)


def update_user(user_id: int, patch: dict) -> dict | None:
    """按 PATCH 语义更新;返回更新后的用户,不存在返回 None。"""
    allowed: dict = {}
    if "password" in patch and patch["password"]:
        allowed["password_hash"] = _hash_password(patch["password"])
    if "display_name" in patch:
        allowed["display_name"] = patch["display_name"]
    if "department" in patch:
        allowed["department"] = patch["department"]
    if "role" in patch and patch["role"] in ROLES:
        allowed["role"] = patch["role"]
    if "status" in patch and patch["status"] in (0, 1):
        allowed["status"] = patch["status"]
    if allowed:
        with engine().begin() as conn:
            conn.execute(update(users).where(users.c.id == user_id).values(**allowed))
    return get_user(user_id)


# ---------------------------------------------------------------- 知识库

def _row_kb(row) -> dict:
    d = dict(row._mapping)
    # retrieval_config 为 TEXT 存的 JSON;对外暴露解析后的 retrieval 字典
    try:
        d["retrieval"] = json.loads(d.pop("retrieval_config") or "{}")
    except (json.JSONDecodeError, TypeError):
        d["retrieval"] = {}
    return d


_KB_CACHE_TTL = 2.0     # 短 TTL:同请求内多图节点共享;跨实例一致性由 TTL 兜底


def get_kb(kb_id: int) -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(
            select(knowledge_bases).where(knowledge_bases.c.id == kb_id)).first()
    return _row_kb(row) if row else None


def get_kb_cached(kb_id: int) -> dict | None:
    """短 TTL 缓存版 get_kb:RAG 图一次请求内多个节点都要读 kb 配置,避免每请求重复查库。
    缓存后端走 cache.py:standalone 进程内,cluster 自动共享 Redis;
    配置变更经 update_kb/delete_kb 主动失效。"""
    from . import cache
    kb = cache.kv_get(f"kb:{kb_id}")
    if kb is None:
        kb = get_kb(kb_id)
        if kb:
            cache.kv_set(f"kb:{kb_id}", kb, _KB_CACHE_TTL)
    return kb


def invalidate_kb_cache(kb_id: int) -> None:
    from . import cache
    cache.kv_delete(f"kb:{kb_id}")


def list_kbs() -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(select(knowledge_bases).order_by(knowledge_bases.c.id)).all()
    return [_row_kb(r) for r in rows]


def create_kb(name: str, description: str, visibility: str, owner_id: int) -> dict:
    if visibility not in ("public", "private"):
        visibility = "private"
    with engine().begin() as conn:
        row = conn.execute(
            insert(knowledge_bases).values(
                name=name, description=description, visibility=visibility,
                owner_id=owner_id, created_at=time.time())
            .returning(knowledge_bases.c.id)).first()
        kb_id = row[0]
    return get_kb(kb_id)


_KB_PATCH_COLUMNS = ("name", "description", "visibility", "doc_name",
                     "strategy", "chunk_count", "built_at")


def update_kb(kb_id: int, patch: dict) -> None:
    allowed = {k: v for k, v in patch.items() if k in _KB_PATCH_COLUMNS}
    if "visibility" in allowed and allowed["visibility"] not in ("public", "private"):
        allowed.pop("visibility")
    if "retrieval" in patch and isinstance(patch["retrieval"], dict):
        # main.py 已合并 cur+patch;此处整体覆盖,保证「删除键」(如清空超范围词表)能生效
        allowed["retrieval_config"] = json.dumps(
            patch["retrieval"], ensure_ascii=False)
    if allowed:
        with engine().begin() as conn:
            conn.execute(update(knowledge_bases)
                         .where(knowledge_bases.c.id == kb_id).values(**allowed))
    invalidate_kb_cache(kb_id)


def delete_kb(kb_id: int) -> None:
    """删除知识库及授权/意图/文档记录(同一事务);chat_sessions 保留作历史。"""
    with engine().begin() as conn:
        for t in (kb_grants, kb_intents, documents):
            conn.execute(delete(t).where(t.c.kb_id == kb_id))
        conn.execute(delete(knowledge_bases).where(knowledge_bases.c.id == kb_id))
    invalidate_kb_cache(kb_id)


# ---------------------------------------------------------------- 授权

def _row_grant(row) -> dict:
    return dict(row._mapping)


def list_grants(kb_id: int) -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(select(kb_grants).where(kb_grants.c.kb_id == kb_id)
                            .order_by(kb_grants.c.id)).all()
    return [_row_grant(r) for r in rows]


def add_grant(kb_id: int, target_type: str, target_value: str) -> dict:
    if target_type not in ("user", "department"):
        raise ValueError("授权类型仅支持 user / department")
    if not target_value:
        raise ValueError("授权对象不能为空")
    if target_type == "user":
        with engine().connect() as conn:
            exists = conn.execute(
                select(users.c.id).where(users.c.username == target_value)).first()
        if not exists:
            raise ValueError(f"用户不存在: {target_value}")
    with engine().begin() as conn:
        conn.execute(_insert_ignore(kb_grants).values(
            kb_id=kb_id, target_type=target_type, target_value=target_value,
            created_at=time.time()))
        row = conn.execute(
            select(kb_grants).where(kb_grants.c.kb_id == kb_id,
                                    kb_grants.c.target_type == target_type,
                                    kb_grants.c.target_value == target_value)).first()
    return _row_grant(row)


def remove_grant(grant_id: int) -> None:
    with engine().begin() as conn:
        conn.execute(delete(kb_grants).where(kb_grants.c.id == grant_id))


# ---------------------------------------------------------------- 业务系统数据对接

_CONNECTOR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
CONNECTOR_TYPES = ("rest", "database")


def _row_connector(row) -> dict:
    d = dict(row._mapping)
    try:
        d["config"] = json.loads(d.get("config") or "{}")
    except json.JSONDecodeError:
        d["config"] = {}
    d["enabled"] = bool(d.get("enabled"))
    return d


def list_connectors(enabled_only: bool = False) -> list[dict]:
    stmt = select(connectors).order_by(connectors.c.id)
    if enabled_only:
        stmt = select(connectors).where(connectors.c.enabled == 1) \
                                 .order_by(connectors.c.id)
    with engine().connect() as conn:
        return [_row_connector(r) for r in conn.execute(stmt).all()]


def get_connector(connector_id: int) -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(
            select(connectors).where(connectors.c.id == connector_id)).first()
    return _row_connector(row) if row else None


def create_connector(name: str, label: str, description: str, ctype: str,
                     config_dict: dict, enabled: bool, created_by: str) -> dict | None:
    """名称冲突返回 None;合法性与预览由路由层/connectors.py 先行校验。"""
    name = (name or "").strip().lower()
    if not _CONNECTOR_NAME_RE.fullmatch(name) or ctype not in CONNECTOR_TYPES:
        raise ValueError("连接器名称需为小写字母开头的英文标识,类型仅支持 rest/database")
    try:
        with engine().begin() as conn:
            row = conn.execute(
                insert(connectors).values(
                    name=name, label=(label or name).strip()[:64],
                    description=(description or "")[:500], type=ctype,
                    config=json.dumps(config_dict or {}, ensure_ascii=False),
                    enabled=1 if enabled else 0,
                    created_by=created_by or "", created_at=time.time())
                .returning(connectors.c.id)).first()
    except IntegrityError:
        return None
    return get_connector(row[0])


_CONNECTOR_PATCH_COLUMNS = ("label", "description", "enabled")


def update_connector(connector_id: int, patch: dict) -> None:
    allowed = {k: v for k, v in patch.items() if k in _CONNECTOR_PATCH_COLUMNS}
    if "enabled" in allowed:
        allowed["enabled"] = 1 if allowed["enabled"] else 0
    if "name" in patch:   # 改名:仅在校验通过时生效(唯一冲突抛 IntegrityError)
        name = str(patch["name"]).strip().lower()
        if _CONNECTOR_NAME_RE.fullmatch(name):
            allowed["name"] = name
    if isinstance(patch.get("config"), dict):
        allowed["config"] = json.dumps(patch["config"], ensure_ascii=False)
    if allowed:
        with engine().begin() as conn:
            conn.execute(update(connectors)
                         .where(connectors.c.id == connector_id).values(**allowed))


def delete_connector(connector_id: int) -> None:
    with engine().begin() as conn:
        conn.execute(delete(connectors).where(connectors.c.id == connector_id))


# ---------------------------------------------------------------- 意图配置

_INTENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# 全局内置路由名(kb_id=0 的种子),kb 级意图占用会遮蔽全局路由,禁止
_INTENT_RESERVED = {"chitchat", "out_of_scope"}


def _clean_str_list(v, cap: int) -> list[str]:
    if not isinstance(v, list):
        return []
    seen, out = set(), []
    for s in v:
        if isinstance(s, str):
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out[:cap]


def _row_intent(row) -> dict:
    d = dict(row._mapping)
    d["kb_id"] = d.pop("kb_id")
    try:
        d["utterances"] = json.loads(d["utterances"] or "[]")
        d["boost_keywords"] = json.loads(d["boost_keywords"] or "[]")
    except (json.JSONDecodeError, TypeError):
        d["utterances"], d["boost_keywords"] = [], []
    return d


def list_intents(kb_id: int, include_global: bool = False) -> list[dict]:
    """kb 级意图排在全局(kb_id=0)之前:router 按序匹配,kb 级可覆盖全局同名配置。"""
    cond = kb_intents.c.kb_id == kb_id
    if include_global:
        cond = cond | (kb_intents.c.kb_id == 0)
    with engine().connect() as conn:
        rows = conn.execute(
            select(kb_intents).where(cond)
            .order_by(kb_intents.c.kb_id.desc(), kb_intents.c.id)).all()
    return [_row_intent(r) for r in rows]


def add_intent(kb_id: int, name: str, label: str,
               utterances: list, boost_keywords: list) -> dict:
    name = (name or "").strip()
    if not _INTENT_NAME_RE.match(name):
        raise ValueError("意图名需为小写字母/数字/下划线,且以字母开头")
    if name in _INTENT_RESERVED:
        raise ValueError(f"{name} 为内置路由名,不可占用")
    utts = _clean_str_list(utterances, 20)
    if not utts:
        raise ValueError("至少配置一条示例问法")
    kws = _clean_str_list(boost_keywords, 30)
    try:
        with engine().begin() as conn:
            conn.execute(insert(kb_intents).values(
                kb_id=kb_id, name=name, label=label or name,
                utterances=json.dumps(utts, ensure_ascii=False),
                boost_keywords=json.dumps(kws, ensure_ascii=False),
                created_at=time.time()))
    except IntegrityError:
        raise ValueError(f"意图 {name} 已存在")
    with engine().connect() as conn:
        row = conn.execute(select(kb_intents).where(
            kb_intents.c.kb_id == kb_id, kb_intents.c.name == name)).first()
    return _row_intent(row)


def update_intent(intent_id: int, patch: dict) -> None:
    allowed: dict = {}
    if "label" in patch:
        allowed["label"] = str(patch["label"])
    if "utterances" in patch:
        utts = _clean_str_list(patch["utterances"], 20)
        if not utts:
            raise ValueError("示例问法不能为空")
        allowed["utterances"] = json.dumps(utts, ensure_ascii=False)
    if "boost_keywords" in patch:
        allowed["boost_keywords"] = json.dumps(
            _clean_str_list(patch["boost_keywords"], 30), ensure_ascii=False)
    if not allowed:
        return
    with engine().begin() as conn:
        res = conn.execute(update(kb_intents)
                           .where(kb_intents.c.id == intent_id).values(**allowed))
    if res.rowcount == 0:
        raise ValueError("意图不存在")


def remove_intent(intent_id: int) -> None:
    with engine().begin() as conn:
        conn.execute(delete(kb_intents).where(kb_intents.c.id == intent_id))


# ---------------------------------------------------------------- 会话元数据

def upsert_chat_session(user_id: int, kb_id: int, session_id: str,
                        title: str = "") -> int:
    """刷新会话列表元数据:不存在则插入(turns=1),存在则 turns+1 并更新 updated_at。

    turns=turns+1 在数据库侧原子完成(ON CONFLICT DO UPDATE),避免多实例
    「读-改-写」竞态。返回最新轮数;失败返回 0(不影响主流程)。
    """
    now = time.time()
    title = (title or "").strip()[:60]
    values = dict(user_id=user_id, kb_id=kb_id, session_id=session_id,
                  title=title, turns=1, updated_at=now)
    set_ = {"turns": chat_sessions.c.turns + 1, "updated_at": now}
    conflict = [chat_sessions.c.user_id, chat_sessions.c.kb_id,
                chat_sessions.c.session_id]
    try:
        if _is_pg():
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(chat_sessions).values(**values) \
                .on_conflict_do_update(index_elements=conflict, set_=set_) \
                .returning(chat_sessions.c.turns)
        else:
            from sqlalchemy.dialects.sqlite import insert as lite_insert
            stmt = lite_insert(chat_sessions).values(**values) \
                .on_conflict_do_update(index_elements=conflict, set_=set_) \
                .returning(chat_sessions.c.turns)
        with engine().begin() as conn:
            row = conn.execute(stmt).first()
        return int(row[0]) if row else 0
    except Exception as e:
        logger.warning("chat_sessions upsert 失败(不影响主流程): %s", e)
        return 0


def list_chat_sessions(user_id: int, kb_id: int) -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(
            select(chat_sessions)
            .where(chat_sessions.c.user_id == user_id,
                   chat_sessions.c.kb_id == kb_id)
            .order_by(chat_sessions.c.updated_at.desc()).limit(100)).all()
    return [dict(r._mapping) for r in rows]


def get_chat_session_turns(user_id: int, kb_id: int, session_id: str) -> int:
    with engine().connect() as conn:
        n = conn.execute(
            select(chat_sessions.c.turns).where(
                chat_sessions.c.user_id == user_id,
                chat_sessions.c.kb_id == kb_id,
                chat_sessions.c.session_id == session_id)).scalar()
    return int(n or 0)


def delete_chat_session(user_id: int, kb_id: int, session_id: str) -> None:
    with engine().begin() as conn:
        conn.execute(delete(chat_sessions).where(
            chat_sessions.c.user_id == user_id,
            chat_sessions.c.kb_id == kb_id,
            chat_sessions.c.session_id == session_id))


# ---------------------------------------------------------------- 访问控制

def _grant_hit(user: dict, kb_id: int) -> bool:
    with engine().connect() as conn:
        row = conn.execute(
            select(kb_grants.c.id).where(
                kb_grants.c.kb_id == kb_id,
                ((kb_grants.c.target_type == "user")
                 & (kb_grants.c.target_value == user["username"]))
                | ((kb_grants.c.target_type == "department")
                   & (kb_grants.c.target_value == user["department"]))
            )).first()
    return row is not None


def can_access_kb(user: dict, kb: dict) -> bool:
    """优先级:管理员角色 > 属主 > 公开可见 > 用户级/部门级授权。"""
    if user["role"] in ("admin", "workspace_admin"):
        return True
    if kb.get("owner_id") == user["id"]:
        return True
    if kb.get("visibility") == "public":
        return True
    return _grant_hit(user, kb["id"])


def can_manage_kb(user: dict, kb: dict) -> bool:
    if user["role"] == "admin":
        return True
    return user["role"] == "workspace_admin" and kb.get("owner_id") == user["id"]


def accessible_kbs(user: dict) -> list[dict]:
    if user["role"] in ("admin", "workspace_admin"):
        return list_kbs()
    with engine().connect() as conn:
        granted = select(kb_grants.c.kb_id).where(
            ((kb_grants.c.target_type == "user")
             & (kb_grants.c.target_value == user["username"]))
            | ((kb_grants.c.target_type == "department")
               & (kb_grants.c.target_value == user["department"])))
        rows = conn.execute(
            select(knowledge_bases).where(
                (knowledge_bases.c.visibility == "public")
                | (knowledge_bases.c.owner_id == user["id"])
                | (knowledge_bases.c.id.in_(granted)))
            .order_by(knowledge_bases.c.id)).all()
    return [_row_kb(r) for r in rows]


# ---------------------------------------------------------------- 文档台账

def add_document(kb_id: int, filename: str, pages: int, chars: int,
                 tables: int, uploaded_by: str) -> int | None:
    """登记文档元数据(路线二:文档与 Milvus 向量一致性核对的业务侧锚点)。"""
    try:
        with engine().begin() as conn:
            row = conn.execute(
                insert(documents).values(
                    kb_id=kb_id, filename=filename, pages=pages, chars=chars,
                    tables_count=tables, uploaded_by=uploaded_by,
                    created_at=time.time()).returning(documents.c.id)).first()
        return int(row[0]) if row else None
    except Exception as e:
        logger.warning("文档台账写入失败: %s", e)
        return None


def list_documents(kb_id: int) -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(select(documents).where(documents.c.kb_id == kb_id)
                            .order_by(documents.c.id)).all()
    return [dict(r._mapping) for r in rows]


def delete_document(doc_id: int) -> None:
    with engine().begin() as conn:
        conn.execute(delete(documents).where(documents.c.id == doc_id))


# ---------------------------------------------------------------- 审计日志

def audit_log(user_id: int | None, username: str, action: str, detail: str = "") -> None:
    """操作审计落库(敏感词拦截/授权变更等);失败只告警,不打断业务。"""
    try:
        with engine().begin() as conn:
            conn.execute(insert(audit_logs).values(
                ts=time.time(), user_id=user_id or 0, username=username or "",
                action=action, detail=(detail or "")[:2000]))
    except Exception as e:
        logger.warning("审计日志写入失败: %s", e)


def list_audit_logs(limit: int = 200, action: str | None = None) -> list[dict]:
    stmt = select(audit_logs).order_by(audit_logs.c.id.desc()).limit(limit)
    if action:
        stmt = select(audit_logs).where(audit_logs.c.action == action) \
                   .order_by(audit_logs.c.id.desc()).limit(limit)
    with engine().connect() as conn:
        rows = conn.execute(stmt).all()
    return [dict(r._mapping) for r in rows]
