"""缓存/锁/限流双后端(路线二 第三步)。

- standalone:进程内字典后端(零依赖),单实例语义不变;
- cluster:REDIS_URL 配置后切换 Redis 后端,缓存/锁/限流均为跨实例分布式语义。

提供四类能力:
  1. 通用 KV(带 TTL):kb 配置缓存、语义路由的示例句向量缓存、embedding 缓存;
  2. 分布式锁:索引构建互斥(多实例防止重复构建);
  3. 滑动窗口限流:/api/chat 按用户限流(多实例共享计数);
  4. 进度存取:Celery 任务进度,供前端轮询。
"""
from __future__ import annotations

import logging
import pickle
import threading
import time
import uuid

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- 后端选择

_redis = None
_redis_lock = threading.Lock()
_backend_warned = False


def _backend():
    """惰性返回 Redis 客户端;未配置 REDIS_URL 返回 None(进程内后端)。"""
    global _redis, _backend_warned
    if not config.REDIS_URL:
        return None
    with _redis_lock:
        if _redis is None:
            try:
                import redis
                _redis = redis.Redis.from_url(config.REDIS_URL, socket_timeout=5,
                                              socket_connect_timeout=5)
                _redis.ping()
                logger.info("Redis 后端就绪")
            except Exception as e:
                # Redis 故障时降级进程内后端并告警,不阻断服务(可用性优先)
                if not _backend_warned:
                    logger.error("Redis 连接失败,降级为进程内缓存(限流/锁退化为单实例语义): %s", e)
                    _backend_warned = True
                return None
        return _redis


def reset_backend_for_test(client) -> None:
    """测试注入:替换 Redis 客户端(如 fakeredis)。"""
    global _redis
    with _redis_lock:
        _redis = client


# ---------------------------------------------------------------- 进程内后端

class _MemStore:
    def __init__(self):
        self._data: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            exp, val = item
            if exp and exp < time.time():
                self._data.pop(key, None)
                return None
            return val

    def set(self, key: str, val: bytes, ttl: float) -> None:
        with self._lock:
            self._data[key] = (time.time() + ttl if ttl else 0.0, val)

    def delete(self, *keys: str) -> None:
        with self._lock:
            for k in keys:
                self._data.pop(k, None)

    def incr(self, key: str, ttl: int) -> int:
        with self._lock:
            # 注意:不能调用 self.get()——非可重入锁会死锁,计数逻辑内联
            item = self._data.get(key)
            exp, raw = item if item else (0.0, None)
            if raw is not None and exp and exp < time.time():
                raw, exp = None, 0.0
            n = int(raw.decode()) + 1 if raw else 1
            if not exp:
                exp = time.time() + ttl   # 固定窗口:保留原窗口剩余 TTL
            self._data[key] = (exp, str(n).encode())
            return n


_mem = _MemStore()

# ---------------------------------------------------------------- 通用 KV

def kv_get(key: str):
    raw = _backend().get(key) if _backend() else _mem.get(key)
    if raw is None:
        return None
    try:
        return pickle.loads(raw)
    except Exception:
        return None


def kv_set(key: str, value, ttl: float = 300) -> None:
    raw = pickle.dumps(value)
    if _backend():
        _backend().set(key, raw, ex=int(ttl) or None)
    else:
        _mem.set(key, raw, ttl)


def kv_delete(*keys: str) -> None:
    if _backend():
        _backend().delete(*keys)
    else:
        _mem.delete(*keys)


# ---------------------------------------------------------------- 分布式锁

class build_lock:
    """索引构建互斥锁。用法: with build_lock(kb_id) as ok: ...
    ok=False 表示未抢到锁(已有实例在构建)。Redis SET NX PX + 令牌释放;
    进程内后端退化为 threading.Lock 非阻塞尝试。"""

    _local_locks: dict[str, threading.Lock] = {}
    _local_guard = threading.Lock()

    def __init__(self, name: str, ttl: int = 1800):
        self.name = name
        self.ttl = ttl
        self.token = uuid.uuid4().hex
        self.acquired = False
        self._local: threading.Lock | None = None

    def __enter__(self) -> bool:
        r = _backend()
        if r:
            self.acquired = bool(r.set(f"lock:{self.name}", self.token,
                                       nx=True, ex=self.ttl))
        else:
            with self._local_guard:
                self._local = self._local_locks.setdefault(self.name, threading.Lock())
            self.acquired = self._local.acquire(blocking=False)
        return self.acquired

    def __exit__(self, *exc):
        r = _backend()
        if r:
            if self.acquired:
                # 校验令牌再删,避免误删他实例的锁
                key = f"lock:{self.name}"
                try:
                    if r.get(key) == self.token.encode():
                        r.delete(key)
                except Exception:
                    pass
        elif self._local and self.acquired:
            self._local.release()
        return False


# ---------------------------------------------------------------- 限流(固定窗口)

def rate_limit_ok(name: str, limit: int, window_sec: int = 60) -> bool:
    """固定窗口计数限流:window 内第 limit+1 次起拒绝。多实例经 Redis 共享计数。"""
    key = f"rl:{name}:{int(time.time()) // window_sec}"
    r = _backend()
    if r:
        try:
            pipe = r.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_sec * 2)
            n, _ = pipe.execute()
        except Exception:
            return True     # Redis 故障时放行(可用性优先),并已在 _backend 告警
    else:
        n = _mem.incr(key, window_sec * 2)
    return int(n) <= limit


# ---------------------------------------------------------------- 任务进度

def progress_set(task_key: str, stage: str, percent: int, message: str = "",
                 ttl: int = 3600) -> None:
    kv_set(f"prog:{task_key}",
           {"stage": stage, "percent": percent, "message": message, "ts": time.time()}, ttl)


def progress_get(task_key: str) -> dict | None:
    return kv_get(f"prog:{task_key}")
