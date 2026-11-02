"""The routing engine: the JAH cascade policy, fail-safe, retries and session state."""

from __future__ import annotations

import asyncio

import pytest
from litellm_jah.config import JAHCfg
from litellm_jah.engine import JAHRouter

from conftest import FakeBackend, decision, msgs


def router(p_hard=0.3, **cfg):
    be = FakeBackend(lambda n, s, q: decision(p_hard))
    return JAHRouter(JAHCfg(decision={"type": "rules"}, **cfg), backend=be), be


def test_first_step_is_strong_without_asking(run):
    async def go():
        r, be = router()
        res = await r.decide(msgs(0))
        assert res.tier == "strong" and res.reason == "first_steps" and res.p_hard is None
        assert be.calls == 0
        await r.close()
    run(go())


def test_cascade_hard_escalates_to_strong(run):
    async def go():
        r, be = router(p_hard=0.9)
        res = await r.decide(msgs(1))
        assert res.tier == "strong" and res.reason == "hard" and res.p_hard == 0.9
        assert res.decision["model"] == "fake" and res.decision["state_chars"] > 0
        await r.close()
    run(go())


def test_cascade_routine_goes_fast(run):
    async def go():
        r, be = router(p_hard=0.3)
        res = await r.decide(msgs(1))
        assert res.tier == "fast" and res.reason == "routine"
        await r.close()
    run(go())


def test_stick_keeps_strong_after_an_escalation(run):
    ps = [0.9, 0.1]

    async def go():
        be = FakeBackend(lambda n, s, q: decision(ps[min(n - 1, len(ps) - 1)]))
        r = JAHRouter(JAHCfg(decision={"type": "rules"}, stick_strong=2), backend=be)
        assert (await r.decide(msgs(1))).reason == "hard"  # call 1: escalation, stick_left = 2
        res = await r.decide(msgs(2))
        assert res.tier == "strong" and res.reason == "stick"  # call 2: forced, no decision call
        res = await r.decide(msgs(3))
        assert res.tier == "strong" and res.reason == "stick"
        res = await r.decide(msgs(4))
        assert res.tier == "fast" and res.reason == "routine"  # call 3: the stick ran out
        assert be.calls == 2
        await r.close()
    run(go())


def test_max_fast_streak_forces_strong(run):
    async def go():
        r, be = router(p_hard=0.1, max_fast_streak=2)
        await r.decide(msgs(0))  # first step: strong
        res = await r.decide(msgs(1))
        assert res.tier == "fast"
        res = await r.decide(msgs(2))
        assert res.tier == "fast"
        res = await r.decide(msgs(3))  # two fast steps in a row -> forced strong
        assert res.tier == "strong" and res.reason == "fast_streak"
        await r.close()
    run(go())


def test_mode_off_never_asks(run):
    async def go():
        r, be = router(mode="off")
        res = await r.decide(msgs(1))
        assert res.tier == "strong" and res.reason == "routing_off" and be.calls == 0
        await r.close()
    run(go())


def test_mode_shadow_asks_but_always_strong(run):
    async def go():
        r, be = router(p_hard=0.1, mode="shadow")
        res = await r.decide(msgs(1))
        assert res.tier == "strong" and res.reason == "shadow" and be.calls == 1
        assert res.p_hard == 0.1  # asked and recorded
        await r.close()
    run(go())


def test_fail_safe_on_backend_error(run):
    async def go():
        be = FakeBackend(lambda n, s, q: (_ for _ in ()).throw(RuntimeError("down")))
        r = JAHRouter(JAHCfg(decision={"type": "rules"}), backend=be)
        res = await r.decide(msgs(1))
        assert res.tier == "strong" and res.reason == "decision_error"
        assert "RuntimeError" in res.decision["error"]
        r2 = JAHRouter(JAHCfg(decision={"type": "rules"}, on_error="fast"), backend=be)
        res = await r2.decide(msgs(1))
        assert res.tier == "fast" and res.reason == "decision_error"
        await r.close()
        await r2.close()
    run(go())


def test_fail_safe_on_backend_timeout(run):
    async def go():
        import asyncio

        async def slow(n, s, q):
            await asyncio.sleep(5)
            return decision(0.1)

        r = JAHRouter(JAHCfg(decision={"type": "rules"}, decision_timeout_s=0.05),
                      backend=FakeBackend(slow))
        res = await r.decide(msgs(1))
        assert res.tier == "strong" and res.reason == "decision_error"
        await r.close()
    run(go())


def test_a_retry_of_the_same_step_reuses_the_decision(run):
    async def go():
        r, be = router(p_hard=0.3)
        m = msgs(1)
        res1 = await r.decide(m)
        res2 = await r.decide(m)
        assert res1.tier == res2.tier == "fast"
        assert res2.reason == "repeat:routine"
        assert be.calls == 1
        await r.close()
    run(go())


def test_concurrent_decides_of_a_new_step_decide_once(run):
    # the complexity router may ask for the same step twice at once: one decision call, one result
    async def go():
        import asyncio

        async def slow(n, s, q):
            await asyncio.sleep(0.02)
            return decision(0.3)

        be = FakeBackend(slow)
        r = JAHRouter(JAHCfg(decision={"type": "rules"}, log_dir=None), backend=be)
        m = msgs(1)
        res1, res2 = await asyncio.gather(r.decide(m), r.decide(m))
        assert res1.tier == res2.tier == "fast" and res1.reason == res2.reason == "routine"
        assert be.calls == 1 # the second caller awaited the first, it did not ask again
        await r.close()
    run(go())


def test_deepcopying_a_plugin_shares_its_engine():
    # the proxy deep-copies its config, which carries the resolved plugin instances
    # (e.g. _ConfigWithBaseline): a copy must not fork the engine or it duplicates
    # the sessions, the journal and the decision rows while the hooks keep the original
    import copy

    from litellm_jah.classifier import JahClassifier, jah_classifier, jah_routing_plugin

    plug = JahClassifier(cfg=JAHCfg(decision={"type": "rules"}))
    clone = copy.deepcopy(plug)
    assert clone is not plug and clone.router is plug.router and clone.cfg is plug.cfg
    # both wiring styles of the same module share one engine from the start
    assert jah_routing_plugin.router is jah_classifier.router


def test_old_decided_records_are_pruned(run):
    from litellm_jah.state import session_key

    async def go():
        r, be = router(p_hard=0.1)
        for s in range(1, 11):
            await r.decide(msgs(s))
        sess = await r.store.get(session_key(msgs(10)))
        assert set(sess.decided) == {6, 7, 8, 9, 10}  # steps older than step-4 are dropped
        await r.close()
    run(go())


def test_rules_backend_drives_the_cascade(run):
    async def go():
        r = JAHRouter(JAHCfg(decision={"type": "rules"}))
        res = await r.decide(msgs(1))  # clean tool results -> routine
        assert res.tier == "fast" and res.decision["model"] == "rules"
        await r.close()
    run(go())


def test_rules_backend_escapes_on_an_error_in_the_last_result(run):
    async def go():
        r = JAHRouter(JAHCfg(decision={"type": "rules"}))
        m = msgs(1)
        m[-1]["content"] = "Traceback (most recent call last):\n  File x"
        res = await r.decide(m)
        assert res.tier == "strong" and res.reason == "hard" and res.decision["why"] == "error_signal"
        await r.close()
    run(go())


def test_a_cancelled_decider_does_not_cancel_a_concurrent_waiter(run):
    async def hang():
        await asyncio.sleep(30)

    async def go():
        be = FakeBackend(lambda n, s, q: decision(0.9) if n >= 2 else hang())
        eng = JAHRouter(JAHCfg(), backend=be)
        ms = msgs(1)
        taskA = asyncio.ensure_future(eng.decide(ms, {"session_id": "f4"}))
        taskB = asyncio.ensure_future(eng.decide(ms, {"session_id": "f4"}))
        await asyncio.sleep(0.1) # A is in the (hung) decision, B waits on A's shared future
        taskA.cancel()
        with pytest.raises(asyncio.CancelledError):
            await taskA
        res = await asyncio.wait_for(taskB, timeout=2)
        assert res.tier == "strong" # B decided on its own (script entry 2), no spurious cancellation
        await eng.close()
    run(go())


def test_guard_mode_routes_to_fast_without_asking(run):
    async def go():
        be = FakeBackend(lambda n, s, q: decision(0.9))
        eng = JAHRouter(JAHCfg(mode="guard"), backend=be)
        first = await eng.decide(msgs(0), {"session_id": "g"})
        assert first.tier == "strong" and first.reason == "first_steps" # the opening step is still forced
        second = await eng.decide(msgs(1), {"session_id": "g"})
        assert second.tier == "fast" and second.reason == "guard"
        assert be.calls == 0 # the guard never asks the decision model pre-call
        await eng.close()
    run(go())


def test_a_decisions_override_turns_guard_on(run):
    async def go():
        be = FakeBackend(lambda n, s, q: decision(0.9))
        eng = JAHRouter(JAHCfg(decisions={"routing": {"mode": "guard"}}), backend=be)
        res = await eng.decide(msgs(1), {"session_id": "g2"})
        assert res.tier == "fast" and res.reason == "guard"
        assert be.calls == 0  # the guard never asks the decision model
        await eng.close()
    run(go())
