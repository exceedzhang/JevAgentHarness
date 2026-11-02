"""The in-memory session store: TTL and LRU eviction."""

from __future__ import annotations

import asyncio
import time

from litellm_jah.sessions import InMemorySessionStore, Session


def test_get_creates_and_reuses(run):
    async def go():
        st = InMemorySessionStore()
        s1 = await st.get("a")
        s1.decided[0] = {"tier": "strong"}
        s2 = await st.get("a")
        assert s2 is s1 and s2.decided[0]["tier"] == "strong"
    run(go())


def test_ttl_expires_an_idle_session(run):
    async def go():
        st = InMemorySessionStore(ttl_s=3600)
        s1 = await st.get("a")
        s1.decided[0] = {"tier": "strong"}
        s1.last_seen = time.time() - 7200
        s2 = await st.get("a")
        assert s2 is not s1 and s2.decided == {}
    run(go())


def test_lru_eviction(run):
    async def go():
        st = InMemorySessionStore(max_sessions=3)
        for k in ("a", "b", "c"):
            await st.get(k)
        await st.get("d")  # evicts the least recently used, "a"
        assert list(st._d) == ["b", "c", "d"]
        a1 = await st.get("a")  # "a" is fresh again; evicts "b"
        assert a1.decided == {}
        assert list(st._d) == ["c", "d", "a"]
    run(go())


def test_lru_skips_in_flight_sessions(run):
    async def go():
        st = InMemorySessionStore(max_sessions=3)
        loop = asyncio.get_running_loop()
        a, b, c = await st.get("a"), await st.get("b"), await st.get("c")
        a.pending["stop_trigger"] = loop.create_future()
        c.pending["stop_trigger"] = loop.create_future()
        # b stays idle: over capacity, it is the LRU victim; a and c are in flight and survive
        await st.get("d")
        assert list(st._d) == ["a", "c", "d"]
        assert not a.pending["stop_trigger"].cancelled() and not c.pending["stop_trigger"].cancelled()
        b1 = await st.get("b")  # "b" is fresh again
        assert b1 is not b and b1.pending == {}
    run(go())


def test_evict_cancels_leftover_futures(run):
    async def go():
        st = InMemorySessionStore()
        loop = asyncio.get_running_loop()
        s = Session(key="a")
        f1, f2 = loop.create_future(), loop.create_future()
        s.deciding[1] = f1
        s.pending["stop_trigger"] = f2
        st._evict(s)
        assert f1.cancelled() and f2.cancelled() and s.pending == {} and s.deciding == {}
    run(go())


def test_all_in_flight_overshoots_rather_than_drops_work(run):
    async def go():
        st = InMemorySessionStore(max_sessions=2)
        loop = asyncio.get_running_loop()
        a, b = await st.get("a"), await st.get("b")
        a.pending["stop_trigger"] = loop.create_future()
        b.pending["stop_trigger"] = loop.create_future()
        c = await st.get("c")
        assert c is not None and len(st._d) == 3  # everything is in flight: keep memory high
    run(go())


def test_ttl_keeps_an_in_flight_session(run):
    async def go():
        st = InMemorySessionStore(ttl_s=1.0)
        loop = asyncio.get_running_loop()
        a = await st.get("a")
        a.pending["stop_trigger"] = loop.create_future()
        a.last_seen = time.time() - 2  # the TTL has expired
        a2 = await st.get("a")
        assert a2 is a  # a request is in flight: kept, not replaced
        a.pending = {}
        a.last_seen = time.time() - 2
        a3 = await st.get("a")
        assert a3 is not a and a3.pending == {}  # idle past the TTL: a fresh session
    run(go())
