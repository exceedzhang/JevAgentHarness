"""The pre-call hook: tool select, the stop trigger, and reflection shape the request before the model runs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from litellm_jah.backend import Decision
from litellm_jah.config import JAHCfg
from litellm_jah.context import in_check
from litellm_jah.engine import JAHRouter
from litellm_jah.points import NO_TOOL, REFLECT_NOTE, STOP_TRIGGER_NOTE
from litellm_jah.precall import JahPreCall

from conftest import Scripted, msgs, p_over

TOOLS = [{"type": "function", "function": {"name": n, "description": f"the {n} tool", "parameters": {}}}
         for n in ("bash", "read_file", "web_search")]


def answer(q, chosen, conf):
    return Decision(p=p_over(list(q.options), chosen, conf))


def rd(messages, tools=TOOLS, **extra):
    d = {"model": "jah-auto", "messages": messages, "metadata": {"session_id": "s1"}, "temperature": 0.2}
    if tools is not None:
        d["tools"] = tools
    d.update(extra)
    return d


def run_hook(engine, data, call_type="completion"):
    return JahPreCall(engine=engine).async_pre_call_hook(None, None, data, call_type)


def test_no_decision_points_leaves_the_request_unchanged(run):
    async def main():
        cfg = JAHCfg(decision={"type": "rules"})
        engine = JAHRouter(cfg, backend=Scripted(
            {"next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2})}))
        m = msgs(2)
        out = await run_hook(engine, rd(m))
        assert out == rd(m)
        sess = await engine.store.get("s1")
        assert sess.pending == {} and 2 in sess.decided
        await engine.close()

    run(main())


def test_tool_select_shadow_logs_and_shapes_nothing(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "next_tool": lambda q: answer(q, "read_file", 0.9),
        })
        engine = JAHRouter(JAHCfg(decisions={"tool_select": {"mode": "shadow"}}), backend=backend)
        m = msgs(2)
        out = await run_hook(engine, rd(m))
        assert out == rd(m)  # shadow changes nothing
        sess = await engine.store.get("s1")
        assert set(sess.pending) == {"tool_select"}
        a = await sess.pending["tool_select"]
        assert a.ok and a.mode == "shadow" and a.choice == "read_file"
        assert NO_TOOL in a.options and a.point == "tool_select"
        sess.pending = {}
        await engine.close()

    run(main())


def test_tool_select_cascade_forces_the_tool(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "next_tool": lambda q: answer(q, "read_file", 0.99),
        })
        engine = JAHRouter(JAHCfg(decisions={"tool_select": {"mode": "cascade"}}), backend=backend)
        m = msgs(2)
        out = await run_hook(engine, rd(m))
        assert out["tool_choice"] == {"type": "function", "function": {"name": "read_file"}}
        sess = await engine.store.get("s1")
        assert sess.decided[2]["reason"] == "tool_cascade" and sess.decided[2]["tier"] == "fast"
        assert sess.plan[2]["tool_choice"] == "read_file"
        sess.pending = {}
        await engine.close()

    run(main())


def test_tool_select_cascade_defers_when_unsure(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "next_tool": lambda q: answer(q, "read_file", 0.9),  # below the 0.99 default
        })
        engine = JAHRouter(JAHCfg(decisions={"tool_select": {"mode": "cascade"}}), backend=backend)
        out = await run_hook(engine, rd(msgs(2)))
        assert "tool_choice" not in out
        sess = await engine.store.get("s1")
        assert sess.decided[2]["tier"] == "fast" and sess.decided[2]["reason"] == "routine"
        a = await sess.pending["tool_select"]
        assert a.effect == "defer_unsure" and not a.acted
        sess.pending = {}
        await engine.close()

    run(main())


def test_tool_select_cascade_defers_a_hard_step(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.1, "hard": 0.9}),
            "next_tool": lambda q: answer(q, "read_file", 0.99),
        })
        engine = JAHRouter(JAHCfg(decisions={"tool_select": {"mode": "cascade"}}), backend=backend)
        out = await run_hook(engine, rd(msgs(2)))
        assert "tool_choice" not in out
        sess = await engine.store.get("s1")
        assert sess.decided[2]["tier"] == "strong"
        a = await sess.pending["tool_select"]
        assert a.effect == "defer_hard"
        sess.pending = {}
        await engine.close()

    run(main())


def test_tool_select_cascade_picks_the_strong_tier_when_configured(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "next_tool": lambda q: answer(q, "read_file", 0.99),
        })
        engine = JAHRouter(
            JAHCfg(decisions={"tool_select": {"mode": "cascade", "tier": "strong", "force": False}}),
            backend=backend)
        out = await run_hook(engine, rd(msgs(2)))
        assert "tool_choice" not in out  # force: false only picks the tier
        sess = await engine.store.get("s1")
        assert sess.decided[2]["tier"] == "strong" and sess.decided[2]["reason"] == "tool_cascade"
        a = await sess.pending["tool_select"]
        assert a.effect == "tier"
        sess.pending = {}
        await engine.close()

    run(main())


def test_tool_select_assist_shortlists_the_tools(run):
    async def main():
        many = ["bash", "read_file", "write_file", "web_search", "grep", "glob", "edit", "test",
                "build", "lint", "docs", "notes", "misc"]
        tools = [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
                 for n in many]

        def answer(q):
            p = {o: 0.0125 for o in q.options}
            p["read_file"] = 0.45
            p["write_file"] = 0.40
            return Decision(p=p)

        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "next_tool": answer,
        })
        engine = JAHRouter(JAHCfg(decisions={"tool_select": {"mode": "assist"}}), backend=backend)
        m = msgs(2)
        out = await run_hook(engine, rd(m, tools=tools))
        kept = [t["function"]["name"] for t in out["tools"]]
        assert set(kept) <= set(many) and len(kept) < len(many)
        assert "read_file" in kept and "write_file" in kept and "bash" in kept  # used tools are always kept
        a = await (await engine.store.get("s1")).pending["tool_select"]
        assert a.effect == "shortlist" and a.extra["kept_tools"] == len(kept)
        (await engine.store.get("s1")).pending = {}
        await engine.close()

    run(main())


def test_tool_select_is_not_asked_on_the_opening_step(run):
    async def main():
        backend = Scripted({"next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2})})
        engine = JAHRouter(JAHCfg(decisions={"tool_select": {"mode": "cascade"}}), backend=backend)
        out = await run_hook(engine, rd(msgs(0)))
        assert out == rd(msgs(0))
        assert "next_tool" not in backend.calls
        (await engine.store.get("s1")).pending = {}
        await engine.close()

    run(main())


def test_reflection_trigger_adds_the_note_and_escalates(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "needs_reflection": lambda q: answer(q, "yes", 0.9),
        })
        engine = JAHRouter(JAHCfg(decisions={"reflection": {"mode": "trigger"}}), backend=backend)
        m = msgs(2)
        out = await run_hook(engine, rd(m))
        assert REFLECT_NOTE in out["messages"][-1]["content"]
        sess = await engine.store.get("s1")
        assert sess.decided[2] == {"tier": "strong", "reason": "reflection", "p_hard": 0.2,
                                   "decision": sess.decided[2]["decision"]}
        a = await sess.pending["reflection"]
        assert a.acted and a.effect == "reflect_on_strong"
        sess.pending = {}
        await engine.close()

    run(main())


def test_reflection_below_tau_changes_nothing(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "needs_reflection": lambda q: answer(q, "yes", 0.5),  # below the 0.8 default
        })
        engine = JAHRouter(JAHCfg(decisions={"reflection": {"mode": "trigger"}}), backend=backend)
        m = msgs(2)
        out = await run_hook(engine, rd(m))
        assert out["messages"] == m
        a = await (await engine.store.get("s1")).pending["reflection"]
        assert a.effect == "no"
        (await engine.store.get("s1")).pending = {}
        await engine.close()

    run(main())


def test_stop_trigger_adds_the_note_and_escalates(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "already_done": lambda q: answer(q, "yes", 0.95),
        })
        engine = JAHRouter(JAHCfg(decisions={"stop": {"mode": "trigger"}}), backend=backend)
        m = msgs(2)
        out = await run_hook(engine, rd(m))
        assert STOP_TRIGGER_NOTE in out["messages"][-1]["content"]
        sess = await engine.store.get("s1")
        assert sess.decided[2]["tier"] == "strong" and sess.decided[2]["reason"] == "stop_trigger"
        a = await sess.pending["stop_trigger"]
        assert a.acted and a.effect == "finish_note_on_strong"
        sess.pending = {}
        await engine.close()

    run(main())


def test_a_retried_step_reuses_its_plan(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "next_tool": lambda q: answer(q, "read_file", 0.99),
        })
        engine = JAHRouter(JAHCfg(decisions={"tool_select": {"mode": "cascade"}}), backend=backend)
        m = msgs(2)
        out1 = await run_hook(engine, rd(m))
        n_calls = len(backend.calls)
        out2 = await run_hook(engine, rd(m))
        assert out2 == out1  # the same plan: the same forced tool
        assert out2["tool_choice"]["function"]["name"] == "read_file"
        assert len(backend.calls) == n_calls  # no decision call on the retry
        (await engine.store.get("s1")).pending = {}
        await engine.close()

    run(main())


def test_a_retry_after_an_escalation_goes_out_plain(run):
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "next_tool": lambda q: answer(q, "read_file", 0.9),
        })
        engine = JAHRouter(JAHCfg(decisions={"tool_select": {"mode": "guard"}}), backend=backend)
        m = msgs(2)
        await engine.decide(m, {"session_id": "s1"})
        await JahPreCall(engine=engine).async_pre_call_hook(None, None, rd(m), "completion")
        sess = await engine.store.get("s1")
        assert sess.plan[2]["tier"] == "fast"
        # the first attempt's checks settle (as the post-call hook would) and its redo escalated the step
        await asyncio.gather(*sess.pending.values(), return_exceptions=True)
        sess.pending = {}
        sess.decided[2] = {**sess.decided[2], "tier": "strong", "reason": "stop_guard"}
        out = await JahPreCall(engine=engine).async_pre_call_hook(None, None, rd(m), "completion")
        assert out == rd(m)  # the notes only (there are none): the tool set was re-decided by the redo
        assert sess.pending == {}
        await engine.close()

    run(main())


def test_internal_calls_and_other_call_types_are_untouched(run):
    async def main():
        cfg = JAHCfg(decision={"type": "rules"})
        engine = JAHRouter(cfg, backend=Scripted(
            {"next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2})}))
        m = msgs(2)
        tok = in_check.set(True)
        try:
            assert await run_hook(engine, rd(m)) is None
        finally:
            in_check.reset(tok)
        assert await run_hook(engine, rd(m), call_type="text_completion") is None
        assert await run_hook(engine, {"model": "jah-auto", "messages": []}) is None
        await engine.close()

    run(main())


def test_the_proxy_async_call_type_is_checked(run):
    # the proxy dispatches hooks with CallTypes.acompletion, not "completion"
    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "needs_reflection": lambda q: answer(q, "yes", 0.9),
        })
        engine = JAHRouter(JAHCfg(decisions={"reflection": {"mode": "trigger"}}), backend=backend)
        m = msgs(2)
        async_completion = SimpleNamespace(value="acompletion") # like litellm's CallTypes.acompletion
        out = await run_hook(engine, rd(m), call_type=async_completion)
        assert REFLECT_NOTE in out["messages"][-1]["content"] # the hook ran, not skipped
        await engine.close()

    run(main())


def test_cancellation_clears_the_pending_asks(run):
    async def hang(q):
        await asyncio.Event().wait()

    async def main():
        backend = Scripted({
            "next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2}),
            "already_done": hang,  # the stop trigger question's id
        })
        engine = JAHRouter(JAHCfg(decisions={"stop": {"mode": "trigger"}}), backend=backend)
        m = msgs(2)
        task = asyncio.ensure_future(run_hook(engine, rd(m)))
        for _ in range(200):  # until the blocking ask is in flight
            if (await engine.store.get("s1")).pending:
                break
            await asyncio.sleep(0.01)
        fut = (await engine.store.get("s1")).pending["stop_trigger"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fut.cancelled()  # the in-flight ask was cancelled with the request
        assert (await engine.store.get("s1")).pending == {}
        await engine.close()

    run(main())


def test_other_model_groups_are_untouched(run):
    async def main():
        cfg = JAHCfg(decision={"type": "rules"})
        engine = JAHRouter(cfg, backend=Scripted(
            {"next_step": lambda q: Decision(p={"routine": 0.8, "hard": 0.2})}))
        m = msgs(2)
        out = await run_hook(engine, rd(m, model="somebody-else"))
        assert out is None  # untouched: the hook declines the foreign group
        sess = await engine.store.get("s1")
        assert sess.decided == {} and sess.pending == {}  # never routed
        await engine.close()

    run(main())
