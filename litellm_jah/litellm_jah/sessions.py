"""Session state across the steps of one agent run: what the routing policy carries between requests.

A session is the agent's conversation (same task): its step number, its current tier, its fast streak and its
pending escalations (``stick_left``). Two stores: the in-process LRU with an idle TTL (the default, fine for
one proxy replica), and a Redis-backed shared store for multi-replica proxies and multi-worker ``litellm`` —
``make_store(cfg)`` picks one from the ``session_store`` config block.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .config import JAHCfg


@dataclass
class Session:
    key: str
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_tier: str = ""
    fast_streak: int = 0
    stick_left: int = 0
    decided: dict[int, dict[str, Any]] = field(default_factory=dict) # step -> route record (reused on a retry)
    deciding: dict[int, Any] = field(default_factory=dict) # step -> in-flight Future[RouteResult]
    pending: dict[str, Any] = field(default_factory=dict) # pre-call questions of the current step: name -> Future[Ask]
    plan: dict[int, dict[str, Any]] = field(default_factory=dict) # step -> {tier, reason, notes, tool_choice, keep}
    redo_calls: dict[int, list[dict[str, Any]]] = field(default_factory=dict) # step -> the strong re-run calls
    step_points: dict[int, dict[str, Any]] = field(default_factory=dict) # step -> {cost_usd, n_asks} this step spent


def _busy(s: Session) -> bool:
    return bool(s.deciding) or bool(s.pending)


def _evict(s: Session) -> None:
    # a session is dropped while nothing is in flight; cancel any straggler
    # future so it cannot outlive the session (and hold the event loop)
    for f in list(s.deciding.values()) + list(s.pending.values()):
        if not f.done():
            f.cancel()
    s.deciding, s.pending = {}, {}


class SessionStore:
    async def get(self, key: str) -> Session: # pragma: no cover - interface
        raise NotImplementedError

    async def save(self, s: Session) -> None:
        """Persist the session's durable state (a no-op for the in-process store)."""
        return None

    async def close(self) -> None:
        return None


class InMemorySessionStore(SessionStore):
    """LRU eviction with an idle TTL. Single process: fine for one proxy replica."""

    def __init__(self, max_sessions: int = 5000, ttl_s: float = 3600.0) -> None:
        self._d: OrderedDict[str, Session] = OrderedDict()
        self._lock = asyncio.Lock()
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s

    @staticmethod
    def _busy(s: Session) -> bool:
        return _busy(s)

    @staticmethod
    def _evict(s: Session) -> None:
        _evict(s)

    async def get(self, key: str) -> Session:
        async with self._lock:
            now = time.time()
            s = self._d.get(key)
            if s is not None and now - s.last_seen <= self.ttl_s:
                s.last_seen = now
                self._d.move_to_end(key)
            elif s is not None and _busy(s):
                # idle TTL lapsed but a request is still in flight: keep the session
                # (dropping it would orphan its decided record mid-step)
                s.last_seen = now
                self._d.move_to_end(key)
            else:
                if s is not None:
                    _evict(s)
                s = Session(key=key)
                self._d[key] = s
            while len(self._d) > self.max_sessions:
                victim = None
                for k in list(self._d): # LRU: the least recently used idle session goes first
                    if k != key and not _busy(self._d[k]):
                        victim = k
                        break
                if victim is None:
                    break # everything is in flight: keep memory high rather than drop work
                _evict(self._d.pop(victim))
            return s


# ------------------------------------------------------------------ the shared (Redis) store


# after a store failure, stop retrying for this long and serve process-locally (the store is
# observability state, not the data path: a dead store must never stall a request)
_DEAD_RETRY_S = 30.0


class _StoreDown(Exception):
    pass


def _redis_client(url: str) -> Any:
    """Lazy on purpose: ``redis`` is an optional dependency (the ``redis`` extra), and a proxy should start
    even when the configured store cannot be reached (fail-safe: JAH degrades to a process-local session)."""
    try:
        import redis.asyncio as redis
    except ImportError as e:
        raise ImportError("the redis session store needs the redis package: pip install redis") from e
    # bounded timeouts: a dead or slow store must never stall a request
    return redis.from_url(url, decode_responses=True, socket_connect_timeout=1, socket_timeout=2)


def _dump(s: Session) -> str:
    # the durable fields only: deciding/pending are in-process asyncio futures, never serialized
    return json.dumps(
        {
            "created": s.created, "last_seen": s.last_seen, "last_tier": s.last_tier,
            "fast_streak": s.fast_streak, "stick_left": s.stick_left,
            "decided": s.decided, "plan": s.plan, "redo_calls": s.redo_calls, "step_points": s.step_points,
        }
    )


def _load(s: Session, blob: str) -> None:
    d = json.loads(blob)
    s.created = float(d.get("created") or time.time())
    s.last_seen = float(d.get("last_seen") or time.time())
    s.last_tier = str(d.get("last_tier") or "")
    s.fast_streak = int(d.get("fast_streak") or 0)
    s.stick_left = int(d.get("stick_left") or 0)
    s.decided = {int(k): v for k, v in (d.get("decided") or {}).items()}
    s.plan = {int(k): v for k, v in (d.get("plan") or {}).items()}
    s.redo_calls = {int(k): v for k, v in (d.get("redo_calls") or {}).items()}
    s.step_points = {int(k): v for k, v in (d.get("step_points") or {}).items()}


class RedisSessionStore(SessionStore):
    """A shared session store: one JSON blob per session in Redis, re-written (with a fresh idle TTL) when a
    step settles (see ``postcall.settle``).

    Each process keeps a local LRU cache of the sessions it touched, with the same in-flight protection as
    the in-memory store: concurrent requests in one process share one ``Session`` object, so in-flight
    decisions and pending questions keep working even after the blob went stale; ``deciding`` and ``pending``
    never leave the process. A session read after its idle TTL is the state at its last settle; a
    conversation advances one step at a time, so two processes writing one session concurrently is
    last-writer-wins in practice.
    """

    def __init__(self, url: str, ttl_s: float = 3600.0, prefix: str = "jah:", max_sessions: int = 5000,
                 client: Any = None) -> None:
        self._url = url
        self._client = client # injected in tests; built lazily from url otherwise
        self._prefix = prefix
        self._ex = max(1, int(ttl_s))
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._d: OrderedDict[str, Session] = OrderedDict()
        self._lock = asyncio.Lock()
        self._dead_until = 0.0
        self._fetches: dict[str, asyncio.Future[Session]] = {}  # single-flight: one in-flight fetch per key

    async def _r(self) -> Any:
        if self._client is None:
            self._client = _redis_client(self._url)
        return self._client

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def _remote(self, key: str) -> Session:
        if time.time() < self._dead_until:
            raise _StoreDown()
        r = await self._r()
        blob = await r.get(self._k(key))
        if not blob:
            return Session(key=key)
        s = Session(key=key)
        try:
            _load(s, blob if isinstance(blob, str) else blob.decode())
        except Exception: # noqa: BLE001 - a corrupt blob is worse than a fresh session
            return Session(key=key)
        return s

    async def get(self, key: str) -> Session:
        fetch: asyncio.Future[Session] | None = None
        fetcher = False
        async with self._lock:
            now = time.time()
            s = self._d.get(key)
            if s is not None and (now - s.last_seen <= self.ttl_s or _busy(s)):
                # local hit: same object, so in-flight futures stay shared within this process
                s.last_seen = now
                self._d.move_to_end(key)
                return s
            if s is not None:
                _evict(s) # stale and idle: drop the local copy, cancel any straggler
                del self._d[key]
            # a remote fetch is needed; share one in-flight fetch per key (fetching under the lock
            # would serialize every session's I/O behind a single connection round-trip)
            fetch = self._fetches.get(key)
            if fetch is None:
                fetch = asyncio.get_running_loop().create_future()
                self._fetches[key] = fetch
                fetcher = True
        try:
            if fetcher:
                try:
                    s = await self._remote(key)
                except Exception as e: # noqa: BLE001 - store down: serve process-locally, never break the request
                    s = Session(key=key)
                    self._dead_until = time.time() + _DEAD_RETRY_S
                    if os.environ.get("JAH_DEBUG"):
                        print(f"JAH session store: {type(e).__name__}; serving process-locally "
                              f"for {_DEAD_RETRY_S:.0f}s", file=sys.stderr, flush=True)
                fetch.set_result(s) # the future never fails: every awaiter gets a usable session
        except BaseException:
            # the fetch itself died (cancellation) before it resolved: drop the in-flight record and
            # hand the waiting callers a process-local fallback rather than a never-settling future
            if fetcher:
                async with self._lock:
                    self._fetches.pop(key, None)
                    if not fetch.done():
                        fetch.set_result(Session(key=key))
            raise
        if not fetcher:
            s = await fetch  # ride the in-flight fetch another caller started
        async with self._lock:
            s.last_seen = time.time()
            self._d[key] = s
            self._d.move_to_end(key)
            if fetcher:
                self._fetches.pop(key, None) # install and pop together: no window for a second fetch
            self._evict_lru(key)
            return s

    def _evict_lru(self, keep: str) -> None:
        while len(self._d) > self.max_sessions:
            victim = None
            for k in list(self._d): # LRU: the least recently used idle session goes first
                if k != keep and not _busy(self._d[k]):
                    victim = k
                    break
            if victim is None:
                break # everything is in flight: keep memory high rather than drop work
            _evict(self._d.pop(victim))

    async def save(self, s: Session) -> None:
        if time.time() < self._dead_until:
            return # store down: skip the write, it would only cost a connection timeout
        r = await self._r()
        await r.set(self._k(s.key), _dump(s), ex=self._ex)

    async def close(self) -> None:
        r = self._client
        if r is None:
            return
        for name in ("aclose", "close"): # redis-py >= 5 has aclose, older ones close
            fn = getattr(r, name, None)
            if fn is None:
                continue
            try:
                await fn()
            except Exception: # noqa: BLE001
                pass
            break


_STORE_KEYS: dict[str, set[str]] = {"memory": set(), "redis": {"url", "prefix"}}


def make_store(cfg: JAHCfg) -> SessionStore:
    """The session store from the config: in-memory by default, Redis for a multi-replica / multi-worker
    proxy (``session_store: {type: redis, url: ...}``)."""
    spec = cfg.session_store
    if spec is None:
        return InMemorySessionStore(cfg.max_sessions, cfg.session_ttl_s)
    if not isinstance(spec, Mapping):
        raise ValueError("session_store: must be a mapping, e.g. {type: redis, url: redis://localhost:6379/0}")
    spec = dict(spec)
    kind = spec.pop("type", "memory")
    allowed = _STORE_KEYS.get(kind)
    if allowed is None:
        raise ValueError(f"session_store: unknown type {kind!r}, expected one of {sorted(_STORE_KEYS)}")
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"unknown session_store keys for type {kind!r}: {sorted(unknown)}")
    if kind == "redis":
        url = spec.get("url")
        if not url:
            raise ValueError("session_store: type redis needs a url, e.g. redis://localhost:6379/0")
        return RedisSessionStore(str(url), ttl_s=cfg.session_ttl_s, prefix=str(spec.get("prefix") or "jah:"),
                                 max_sessions=cfg.max_sessions)
    return InMemorySessionStore(cfg.max_sessions, cfg.session_ttl_s)
