"""全局配置与运行时设置管理。

数据目录优先级:环境变量 DATA_DIR > 项目默认 server/data。
DeepSeek 密钥:环境变量 DEEPSEEK_API_KEY 作为初始值,运行时可通过
设置接口修改并持久化到 DATA_DIR/settings.json。

部署档位(DEPLOY_PROFILE):
- standalone(默认,单实例):SQLite + Milvus Lite + 进程内任务与缓存,
  零中间件,docker run 即可跑;
- cluster(路线二·多实例可扩容):IvorySQL(PG 协议)+ 独立 Milvus + Redis
  (缓存/锁/限流/Celery Broker)+ Celery Worker + 可选 TEI 向量服务,
  全部由下方环境变量注入,代码同一套。
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # server/
PROJECT_DIR = BASE_DIR.parent                              # 项目根
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data")).resolve()
SEED_PDF = BASE_DIR / "data_seed" / "employee_handbook.pdf"  # 内置样例文档(员工规章制度手册)
FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", PROJECT_DIR / "dist")).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = DATA_DIR / "settings.json"
MILVUS_FILE = DATA_DIR / "milvus_lite.db"
MEMORY_DB = DATA_DIR / "memory.db"
STATS_DB = DATA_DIR / "stats.db"
USERS_DB = DATA_DIR / "users.db"                           # 账号/权限库(standalone)
KBS_DIR = DATA_DIR / "kbs"                                 # 各知识库数据(文档/缓存/索引元信息)
KBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE = DATA_DIR / "models"                          # embedding 模型缓存


def kb_dir(kb_id: int) -> Path:
    """单个知识库的数据目录:document.json / source.pdf / chunk_cache.pkl / index_meta.json。"""
    d = KBS_DIR / str(kb_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- 部署档位

DEPLOY_PROFILE = os.environ.get("DEPLOY_PROFILE", "standalone")   # standalone | cluster


def is_cluster() -> bool:
    return DEPLOY_PROFILE == "cluster"


# ---- 路线二·中间件配置(cluster 档生效,全部有 standalone 回退) ----

# IvorySQL(PG 协议完全兼容,驱动走 postgresql+psycopg;cluster 必填)
DB_URI = os.environ.get(
    "DB_URI", f"sqlite:///{USERS_DB}")                          # 业务库 rag_db
CHECKPOINT_DB_URI = os.environ.get(
    "CHECKPOINT_DB_URI", "")                                    # LangGraph 会话检查点库 rag_checkpoint
# 多实例同时启动时 checkpointer.setup() 的幂等 DDL 会争抢 migrations 锁;
# 生产在部署阶段一次性建表后,置 1 跳过运行时 setup(standalone 无竞争可保持 0)
SKIP_CHECKPOINTER_SETUP = os.environ.get("SKIP_CHECKPOINTER_SETUP", "0") == "1"
# Redis:缓存 + 分布式锁 + 限流 + 任务进度 + Celery Broker
REDIS_URL = os.environ.get("REDIS_URL", "")
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)
# 独立 Milvus 服务(cluster);standalone 用本地 Milvus Lite 文件
MILVUS_URI = os.environ.get("MILVUS_URI", "")                   # 如 http://milvus:19530
# Embedding:local(web 进程内加载 BGE,方案A)| tei(TEI HTTP 服务,方案B)
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "local")
TEI_URL = os.environ.get("TEI_URL", "")                         # 如 http://tei:8080
# DeepSeek 主备 Key(备用 Key 只在主 Key 调用失败时切换)
DEEPSEEK_BACKUP_API_KEY = os.environ.get("DEEPSEEK_BACKUP_API_KEY", "")
SCAN_OCR = os.getenv("SCAN_OCR", "1") == "1"          # 扫描页 OCR 转录(RapidOCR,离线);置 0 关闭
# 种子数据:内置账号/内置知识库默认开启(开箱即用);生产部署置 SEED_DEMO_DATA=0,
# 只保留 admin 一个账号,初始密码用 ADMIN_PASSWORD 注入并第一时间改密
SEED_DEMO_DATA = os.environ.get("SEED_DEMO_DATA", "1") == "1"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
# CORS:cluster 档收敛为前端域名列表(逗号分隔);standalone 放开 *
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
# 上传限制
UPLOAD_MAX_MB = int(os.environ.get("UPLOAD_MAX_MB", "50"))
UPLOAD_EXTS = {".pdf", ".docx", ".xlsx"}                        # 仅文本类文档,剔除图片/OCR/CAD
# 问答限流(每用户每分钟,Redis 滑动窗口;standalone 用内存)
RATE_LIMIT_CHAT_PER_MIN = int(os.environ.get("RATE_LIMIT_CHAT_PER_MIN", "60"))

DEFAULTS = {
    "deepseek_api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "deepseek_base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    "deepseek_model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    # 视觉模型(图片描述):须支持图片输入,默认 deepseek-flash(V4.1-Flash 原生视觉);
    # 注意 deepseek-v4-pro 等纯文本推理模型不支持图片,填错会导致描述全部失败
    "vision_model": os.environ.get("VISION_MODEL", "deepseek-flash"),
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "memory_turns": 50,        # 短时记忆:前后 50 轮对话
    "retrieve_top_k": 6,
    # 拒答话术:默认确定性模板(不把越界请求喂给 LLM);置 1 时由 LLM 生成,
    # 且只让模型写话术、不回传用户原文与历史(见 agent.py refuse 注释)
    "refusal_llm": os.environ.get("REFUSAL_LLM", "0") == "1",
    # 备用 Key(故障切换):与主 Key 同源(settings.json 可配),环境变量为默认值
    "deepseek_backup_api_key": DEEPSEEK_BACKUP_API_KEY,
    # 工具调用(一期):tool 意图 → LangGraph 工具节点,内置工具 + MCP 扩展。
    # 工具节点固定用非思考模型(deepseek-chat 类)——思考模式的工具多轮
    # 回传 reasoning_content 适配留作二期;检索/闲聊链路模型不受影响。
    "tools_enabled": os.environ.get("TOOLS_ENABLED", "1") == "1",
    "tools_model": os.environ.get("TOOLS_MODEL", "deepseek-chat"),
    # MCP servers(JSON 字符串,设置页可配):
    #   {"amap": {"url": "https://.../sse"}, "local": {"command": "uvx", "args": ["..."]}}
    # 默认空 = 只用内置工具;server 为独立进程,注意选择国内可用的服务
    "mcp_servers": os.environ.get("MCP_SERVERS", ""),
    # 技能(Skills,二期):SKILL.md 指令包,随工具链路的 use_skill 按需加载。
    # skills_disabled 为按名停用名单(内置技能只读,靠停用下线)
    "skills_enabled": os.environ.get("SKILLS_ENABLED", "1") == "1",
    "skills_disabled": [],
}

_lock = threading.Lock()


def load_settings() -> dict:
    with _lock:
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text("utf-8"))
            except Exception:
                data = {}
        else:
            data = {}
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in data.items() if v is not None})
    # 环境变量始终可覆盖密钥,方便容器部署注入
    if os.environ.get("DEEPSEEK_API_KEY"):
        merged["deepseek_api_key"] = os.environ["DEEPSEEK_API_KEY"]
    return merged


def save_settings(patch: dict) -> dict:
    current = load_settings()
    current.update({k: v for k, v in patch.items() if v is not None})
    with _lock:
        SETTINGS_FILE.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), "utf-8"
        )
    return current


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return key[:2] + "****"
    return key[:4] + "****" + key[-4:]
