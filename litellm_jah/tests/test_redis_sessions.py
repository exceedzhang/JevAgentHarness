"""The shared (Redis) session store: round-trips, in-flight protection, make_store, cross-process continuity."""

from __future__ import annotations

import asyncio
import time

import pytest
from litellm_jah.config import JAHCfg
from litellm_jah.engine import JAHRouter
from litellm_jah.sessions import (
    InMemorySessionStore,
    RedisSessionStore,
    Session,
    _evict,
    make_store,
)

from conftest import FakeBackend, decision, msgs


class FakeRedis:
    """A tiny redis.asyncio stand-in (str values, like decode_responses=True)."""

    def __init__(self) -> None:
        self.d: dict[str, str] = {}
        self.expire_at: dict[str, float] = {}
        self.closed = False
        self.down = False

    async def get(self, k: str) -> str | None:
        if self.down:
            raise OSError("connection refused")
        if k in self.expire_at and self.expire_at[k] <= time.time():
            self.d.pop(k, None)
            self.expire_at.pop(k, None)
            return None
        return self.d.get(k)

    async def set(self, k: str, v: str, ex: float | None = None) -> bool:
        self.d[k] = v
        if ex is not None:
            self.expire_at[k] = time.time() + ex
        return True

    async def aclose(self) -> None:
        self.closed = True


def test_save_roundtrip_between_stores(run):
 async def go():
  fake = FakeRedis()
  a, b = RedisSessionStore("redis://x", client=fake), RedisSessionStore("redis://x", client=fake)
  s = await a.get("k")
  s.last_tier = "fast"
  s.fast_streak = 2
  s.stick_left = 1
  s.decided[3] = {"tier": "fast", "reason": "p", "p_hard": 0.2, "decision": {"p": 0.2}}
  s.plan[3] = {"tier": "fast", "reason": "p", "notes": ["n"], "tool_choice": None, "keep": None}
  s.redo_calls[2] = [{"model": "fake", "latency_s": 1.0}]
  s.step_points[2] = {"cost_usd": 0.01, "n_asks": 1}
  await a.save(s)
  s2 = await b.get("k")
  assert s2.last_tier == "fast" and s2.fast_streak == 2 and s2.stick_left == 1
  assert s2.decided[3]["tier"] == "fast"
  assert s2.plan[3]["notes"] == ["n"]
  assert s2.redo_calls[2][0]["model"] == "fake"
  assert s2.step_points[2]["n_asks"] == 1
  assert s2.deciding == {} and s2.pending == {} # in-flight futures never leave the process
 run(go())


def test_get_returns_the_same_object_within_a_process(run):
 async def go():
  st = RedisSessionStore("redis://x", client=FakeRedis())
  s1 = await st.get("k")
  s1.stick_left = 5
  s2 = await st.get("k")
  assert s2 is s1 and s2.stick_left == 5
 run(go())


def test_idle_session_is_reread_from_redis(run):
 async def go():
  fake = FakeRedis()
  st = RedisSessionStore("redis://x", ttl_s=1.0, client=fake)
  s1 = await st.get("k")
  s1.stick_left = 3
  await st.save(s1)
  s1.last_seen = time.time() - 2 # the local idle TTL lapsed
  s2 = await st.get("k")
  assert s2 is not s1 and s2.stick_left == 3 # the state at the last settle
 run(go())


def test_remote_expiry_starts_fresh(run):
 async def go():
  fake = FakeRedis()
  st = RedisSessionStore("redis://x", ttl_s=1.0, client=fake)
  s1 = await st.get("k")
  s1.stick_left = 3
  await st.save(s1)
  fake.expire_at["jah:k"] = time.time() - 1 # the blob expired in redis
  s1.last_seen = time.time() - 2
  s2 = await st.get("k")
  assert s2 is not s1 and s2.stick_left == 0
 run(go())


def test_busy_session_survives_the_local_ttl(run):
 async def go():
  st = RedisSessionStore("redis://x", ttl_s=1.0, client=FakeRedis())
  loop = asyncio.get_running_loop()
  s = await st.get("k")
  s.pending["stop_trigger"] = loop.create_future()
  s.last_seen = time.time() - 2 # the TTL has expired, but a question is in flight
  s2 = await st.get("k")
  assert s2 is s # kept, not replaced: the in-flight future must stay shared
 run(go())


def test_lru_skips_in_flight_sessions(run):
 async def go():
  st = RedisSessionStore("redis://x", client=FakeRedis(), max_sessions=3)
  loop = asyncio.get_running_loop()
  a, b, c = await st.get("a"), await st.get("b"), await st.get("c")
  a.pending["stop_trigger"] = loop.create_future()
  c.pending["stop_trigger"] = loop.create_future()
  # b stays idle: over capacity, it is the LRU victim; a and c are in flight and survive
  await st.get("d")
  assert list(st._d) == ["a", "c", "d"]
  b1 = await st.get("b") # "b" is fresh again
  assert b1 is not b and b1.pending == {}
 run(go())


def test_all_in_flight_overshoots_rather_than_drops_work(run):
 async def go():
  st = RedisSessionStore("redis://x", client=FakeRedis(), max_sessions=2)
  loop = asyncio.get_running_loop()
  a, b = await st.get("a"), await st.get("b")
  a.pending["stop_trigger"] = loop.create_future()
  b.pending["stop_trigger"] = loop.create_future()
  c = await st.get("c")
  assert c is not None and len(st._d) == 3 # everything is in flight: keep memory high
 run(go())


def test_evict_cancels_leftover_futures(run):
 async def go():
  loop = asyncio.get_running_loop()
  s = Session(key="a")
  f1, f2 = loop.create_future(), loop.create_future()
  s.deciding[1] = f1
  s.pending["stop_trigger"] = f2
  _evict(s)
  assert f1.cancelled() and f2.cancelled() and s.pending == {} and s.deciding == {}
 run(go())


def test_deciding_is_not_shared_across_stores(run):
 async def go():
  fake = FakeRedis()
  a, b = RedisSessionStore("redis://x", client=fake), RedisSessionStore("redis://x", client=fake)
  loop = asyncio.get_running_loop()
  s = await a.get("k")
  s.deciding[1] = loop.create_future()
  await a.save(s)
  s2 = await b.get("k")
  assert s2.deciding == {} and s.deciding  # the in-flight future stays in process a
 run(go())


def test_close_closes_redis(run):
 async def go():
  fake = FakeRedis()
  st = RedisSessionStore("redis://x", client=fake)
  await st.close()
  assert fake.closed
 run(go())


def test_make_store(run):
 async def go():
  assert isinstance(make_store(JAHCfg()), InMemorySessionStore)
  assert isinstance(make_store(JAHCfg(session_store={"type": "memory"})), InMemorySessionStore)
  st = make_store(JAHCfg(session_store={"type": "redis", "url": "redis://x"}))
  assert isinstance(st, RedisSessionStore) and st._prefix == "jah:"
  st2 = make_store(JAHCfg(session_store={"type": "redis", "url": "redis://x", "prefix": "p:"}))
  assert st2._prefix == "p:"
  for bad in (
   {"type": "bogus"},
   {"type": "redis"}, # no url
   {"type": "memory", "url": "u"}, # unknown key for memory
   {"type": "redis", "url": "u", "bogus": 1}, # unknown key for redis
   "memory", # not a mapping
  ):
   with pytest.raises(ValueError):
    make_store(JAHCfg(session_store=bad))
 run(go())


def test_config_passes_the_session_store_through():
 cfg = JAHCfg.from_dict({"session_store": {"type": "redis", "url": "redis://localhost:6379/0"}})
 assert cfg.session_store == {"type": "redis", "url": "redis://localhost:6379/0"}


def test_cross_process_continuity(run):
 async def go():
  fake = FakeRedis()
  cfg = JAHCfg()
  rA = JAHRouter(cfg=cfg, backend=FakeBackend(decision(0.9)), store=RedisSessionStore("redis://x", client=fake))
  rB = JAHRouter(cfg=cfg, backend=FakeBackend(decision(0.9)), store=RedisSessionStore("redis://x", client=fake))
  # process a leaves the conversation on an escalation: step 1 must start strong
  sA = await rA.store.get("kp-1")
  sA.stick_left = 2
  sA.last_tier = "fast"
  await rA.store.save(sA)
  res = await rB.decide(msgs(1), {"session_id": "kp-1"})
  assert res.tier == "strong" and res.reason == "stick"
  # process b's step is durable once it settles; process a sees it after a local reload
  sB = await rB.store.get("kp-1")
  await rB.store.save(sB)
  sA.last_seen = time.time() - 7200 # force the local cache to reload from redis
  sA2 = await rA.store.get("kp-1")
  assert sA2 is not sA and sA2.last_tier == "strong" and 1 in sA2.decided
 run(go())

def test_store_down_degrades_to_local_and_retries_later(run):
 async def go():
  st = RedisSessionStore("redis://x", client=FakeRedis())
  st._client.down = True
  s1 = await st.get("a") # store down: a process-local session, no raise
  assert s1 is not None and s1.key == "a" and s1.stick_left == 0
  assert st._dead_until > time.time() # the dead window is open
  s2 = await st.get("a") # local hit: same object
  assert s2 is s1
  s1.stick_left = 5
  await st.save(s1) # skipped while the dead window is open (no write, no connection attempt)
  assert st._client.d == {}
  st._dead_until = 0.0 # the store recovers
  s1.last_seen = time.time() - 7200 # force a reload from redis
  s3 = await st.get("a")
  assert s3 is not s1 # the dead window did not pin the stale local session
 run(go())


def test_concurrent_cold_gets_share_one_remote_fetch(run):
    # single-flight: N concurrent cold gets for one key issue one round-trip, not N, and all
    # callers share the same process-local session object
    async def go():
        fake = FakeRedis()
        n = {"gets": 0}
        orig = fake.get

        async def slow(k):
            n["gets"] += 1
            await asyncio.sleep(0.2)
            return await orig(k)

        fake.get = slow
        st = RedisSessionStore("redis://x", client=fake)
        t0 = time.monotonic()
        results = await asyncio.gather(*(st.get("k") for _ in range(5)))
        dt = time.monotonic() - t0
        assert n["gets"] == 1
        assert all(r is results[0] for r in results)
        assert dt < 0.5  # five serial round-trips would take ~1.0s
        assert st._fetches == {}  # the in-flight record is cleaned up
        run_ok = True
        return run_ok
    run(go())
