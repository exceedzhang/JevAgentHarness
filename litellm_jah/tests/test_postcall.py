"""The post-call hook: the stop check (finish check / guard / shadow), the tool guard, and the action guard,
with their strong-tier re-dos, and the decisions rows they settle."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from litellm_jah.backend import Decision
from litellm_jah.config import JAHCfg
from litellm_jah.context import in_check
from litellm_jah.engine import JAHRouter
from litellm_jah.logger import JahStepLogger
from litellm_jah.postcall import JahPostCall
from litellm_jah.precall import JahPreCall

from conftest import Scripted, msgs, p_over

TOOLS = [{"type": "function", "function": {"name": n, "description": f"the {n} tool", "parameters": {}}}
         for n in ("bash", "read_file")]


def answer(q, chosen, conf):
    return Decision(p=p_over(list(q.options), chosen, conf))


def resp(tool_calls=None, model="g-fast", in_t=100, out_t=20):
    usage = SimpleNamespace(
        prompt_tokens=in_t, completion_tokens=out_t,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
    msg = SimpleNamespace(tool_calls=tool_calls, content=None if tool_calls else "final answer")
    return SimpleNamespace(usage=usage, model=model,
                           choices=[SimpleNamespace(message=msg,
                                                    finish_reason="tool_calls" if tool_calls else "stop")])


def tool(name="bash", args='{"command": "ls"}'):
    return SimpleNamespace(id="tc1", type="function", function=SimpleNamespace(name=name, arguments=args))


def make_engine(answers, decisions, p_hard=0.2, log_dir=None, mode="cascade"):
 backend = Scripted({**{"next_step": lambda q: Decision(p={"routine": 1.0 - p_hard, "hard": p_hard})},
 **answers})
 cfg = JAHCfg(mode=mode, decision={"type": "rules"}, strong="g-strong", decisions=decisions, log_dir=log_dir)
 return JAHRouter(cfg, backend=backend), backend


def rd(messages, tools=TOOLS, **extra):
    # real post-call shape: "model" is the resolved deployment name; the client's group
    # arrives stamped in metadata["model_group"] (the router does this)
    d = {"model": "openai/jah-strong", "messages": messages,
         "metadata": {"session_id": "s1", "model_group": "jah-auto"}, "temperature": 0.2}
    if tools is not None:
        d["tools"] = tools
    d.update(extra)
    return d


def hook(engine, recall=None):
    return JahPostCall(engine=engine, recall=recall)


def decision_rows(tmp):
    return [json.loads(line) for line in Path(tmp, "decisions.jsonl").read_text().splitlines() if line.strip()]


def test_stop_guard_accepts_a_completed_task(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, backend = make_engine(
                {"task_done": lambda q: answer(q, "done", 0.95)}, {"stop": {"mode": "guard"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            out = await hook(engine).async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is None and "task_done" in backend.calls
            rows = [r for r in decision_rows(tmp) if r["point"] == "stop"]
            assert len(rows) == 1 and rows[0]["mode"] == "guard" and rows[0]["effect"] == "accepted"
            assert rows[0]["p_act"] == 0.95
            await engine.close()

    run(main())


def test_stop_guard_redoes_an_unfinished_task(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, backend = make_engine(
                {"task_done": lambda q: answer(q, "not_done", 0.95)}, {"stop": {"mode": "guard"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            seen = []

            async def recall(model, messages, tools, request_data):
                seen.append((model, messages, tools, request_data))
                return resp(tool_calls=[tool()], model="g-strong", in_t=110, out_t=30)

            out = await hook(engine, recall).async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is not None and out.model == "g-strong"
            assert seen and seen[0][0] == "g-strong" and seen[0][1] == m  # the note list was empty on fast tier
            sess = await engine.store.get("s1")
            assert sess.decided[2]["tier"] == "strong" and sess.decided[2]["reason"] == "stop_guard"
            assert [c["tier"] for c in sess.redo_calls[2]] == ["fast", "strong"]
            rows = [r for r in decision_rows(tmp) if r["point"] == "stop"]
            assert rows[0]["effect"] == "redo" and rows[0]["label"] == "not_done"
            assert rows[0]["label_tier"] == "strong" and rows[0]["fit_ok"] is True
            await engine.close()

    run(main())


def test_stop_shadow_behaves_like_strong_and_logs(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine(
                {"task_done": lambda q: answer(q, "not_done", 0.95)}, {"stop": {"mode": "shadow"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})

            async def recall(model, messages, tools, request_data):
                return resp(tool_calls=[tool()], model="g-strong")

            out = await hook(engine, recall).async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is not None and out.model == "g-strong"
            rows = [r for r in decision_rows(tmp) if r["point"] == "stop"]
            assert rows[0]["effect"] == "logged" and rows[0]["label"] == "not_done" and rows[0]["fit_ok"] is True
            await engine.close()

    run(main())


def test_stop_guard_scope_fast_skips_a_strong_step(run):
    async def main():
        engine, backend = make_engine(
            {"task_done": lambda q: answer(q, "done", 0.95)}, {"stop": {"mode": "guard"}}, p_hard=0.9)
        m = msgs(2)
        await engine.decide(m, {"session_id": "s1"})
        out = await hook(engine).async_post_call_success_deployment_hook(rd(m), resp(model="g-strong"), "completion")
        assert out is None and "task_done" not in backend.calls
        await engine.close()

    run(main())


def test_stop_guard_scope_all_checks_a_strong_step(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine(
                {"task_done": lambda q: answer(q, "not_done", 0.95)},
                {"stop": {"mode": "guard", "scope": "all"}}, p_hard=0.9, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})

            async def recall(model, messages, tools, request_data):
                assert "not be complete" in messages[-1]["content"]  # the rejection note on a strong tier
                return resp(tool_calls=[tool("read_file")], model="g-strong")

            out = await hook(engine, recall).async_post_call_success_deployment_hook(rd(m), resp(model="g-strong"),
                                                                                    "completion")
            assert out is not None
            rows = [r for r in decision_rows(tmp) if r["point"] == "stop"]
            assert rows[0]["tier"] == "strong" and rows[0]["effect"] == "redo"
            await engine.close()

    run(main())


def test_tool_guard_redoes_when_the_model_chose_else(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine(
                {"next_tool": lambda q: answer(q, "read_file", 0.99)}, {"tool_select": {"mode": "guard"}},
                log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            await JahPreCall(engine=engine).async_pre_call_hook(None, None, rd(m, model="jah-auto"), "completion")
            seen = []

            async def recall(model, messages, tools, request_data):
                seen.append(messages)
                return resp(tool_calls=[tool("read_file")], model="g-strong")

            out = await hook(engine, recall).async_post_call_success_deployment_hook(
                rd(m), resp(tool_calls=[tool()]), "completion")
            assert out is not None and out.model == "g-strong"
            assert seen and "A check doubts" in seen[0][-1]["content"]
            rows = [r for r in decision_rows(tmp) if r["point"] == "tool_select"]
            assert rows[0]["effect"] == "redo" and rows[0]["label"] == "bash"
            # like the standalone proxy: the guard scored the real reply, so it stays fit_ok=True
            assert rows[0]["choice"] == "read_file" and rows[0]["fit_ok"] is True
            await engine.close()

    run(main())


def test_tool_guard_accepts_when_the_model_agrees(run):
    async def main():
        engine, _ = make_engine(
            {"next_tool": lambda q: answer(q, "bash", 0.99)}, {"tool_select": {"mode": "guard"}})
        m = msgs(2)
        await engine.decide(m, {"session_id": "s1"})
        await JahPreCall(engine=engine).async_pre_call_hook(None, None, rd(m, model="jah-auto"), "completion")
        out = await hook(engine, recall=lambda *a: (_ for _ in ()).throw(AssertionError("no recall"))
                        ).async_post_call_success_deployment_hook(
            rd(m), resp(tool_calls=[tool()]), "completion")
        assert out is None
        sess = await engine.store.get("s1")
        assert sess.pending == {}
        await engine.close()

    run(main())


def test_action_guard_redoes_a_risky_action(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine({"action_risk": lambda q: answer(q, "risky", 0.9)},
                                    {"action": {"mode": "guard"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            seen = []

            async def recall(model, messages, tools, request_data):
                seen.append(messages)
                return resp(tool_calls=[tool("bash", '{"command": "ls"}')], model="g-strong")

            out = await hook(engine, recall).async_post_call_success_deployment_hook(
                rd(m), resp(tool_calls=[tool("bash", '{"command": "rm -rf /tmp/data"}')]), "completion")
            assert out is not None and out.model == "g-strong"
            assert seen and "may be irreversible or risky" in seen[0][-1]["content"]
            rows = [r for r in decision_rows(tmp) if r["point"] == "action"]
            assert rows[0]["effect"] == "recheck" and rows[0]["tool"] == "bash"
            assert rows[0]["recheck"] == "changed" and rows[0]["recheck_risky"] is False
            await engine.close()

    run(main())


def test_action_shadow_logs_without_a_redo(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine({"action_risk": lambda q: answer(q, "risky", 0.9)},
                                    {"action": {"mode": "shadow"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            out = await hook(engine, recall=lambda *a: (_ for _ in ()).throw(AssertionError("no recall"))
                            ).async_post_call_success_deployment_hook(
                rd(m), resp(tool_calls=[tool("bash", '{"command": "rm -rf /tmp/data"}')]), "completion")
            assert out is None
            rows = [r for r in decision_rows(tmp) if r["point"] == "action"]
            assert rows[0]["effect"] == "logged" and rows[0]["pattern"]
            await engine.close()

    run(main())


def test_action_guard_on_error_recheck_redoes(run):
    async def main():
        engine, _ = make_engine(
            {"action_risk": (lambda q: (_ for _ in ()).throw(RuntimeError("down")))},
            {"action": {"mode": "guard", "on_error": "recheck"}})
        m = msgs(2)
        await engine.decide(m, {"session_id": "s1"})
        strong = resp(tool_calls=[tool()], model="g-strong")

        async def recall(model, messages, tools, request_data):
            return strong

        out = await hook(engine, recall).async_post_call_success_deployment_hook(
            rd(m), resp(tool_calls=[tool("bash", '{"command": "rm -rf /tmp/data"}')]), "completion")
        assert out is not None and out is strong
        sess = await engine.store.get("s1")
        assert sess.decided[2]["reason"] == "action_guard"
        await engine.close()

    run(main())


def test_a_risky_redo_gets_one_more_look(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine(
                {"task_done": lambda q: answer(q, "not_done", 0.95), "action_risk": lambda q: answer(q, "risky", 0.9)},
                {"stop": {"mode": "guard"}, "action": {"mode": "guard"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            first = resp(tool_calls=[tool("bash", '{"command": "rm -rf /tmp/data"}')], model="g-strong")
            second = resp(tool_calls=[tool("bash", '{"command": "ls"}')], model="g-strong")
            calls = []

            async def recall(model, messages, tools, request_data):
                calls.append(messages)
                return first if len(calls) == 1 else second

            out = await hook(engine, recall).async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is second and len(calls) == 2
            assert "may be irreversible or risky" in calls[1][-1]["content"]
            sess = await engine.store.get("s1")
            assert [c["tier"] for c in sess.redo_calls[2]] == ["fast", "strong", "strong"]
            # the second look's question is settled too: an action row lands and the step counts both asks
            rows = [r for r in decision_rows(tmp) if r["point"] == "action"]
            assert len(rows) == 1
            assert rows[0]["recheck"] == "changed" and rows[0]["recheck_risky"] is False
            assert sess.step_points[2]["n_asks"] == 2
            await engine.close()

    run(main())


def test_redo_failure_keeps_the_original_reply(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine(
                {"task_done": lambda q: answer(q, "not_done", 0.95)}, {"stop": {"mode": "guard"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})

            async def recall(*a):
                raise RuntimeError("strong tier down")

            out = await hook(engine, recall).async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is None
            sess = await engine.store.get("s1")
            assert sess.decided[2]["tier"] == "fast" and sess.decided[2]["reason"] == "routine"
            assert 2 not in sess.redo_calls
            rows = [r for r in decision_rows(tmp) if r["point"] == "stop"]
            assert rows[0]["effect"] == "error_kept"
            await engine.close()

    run(main())


def test_internal_calls_streams_and_foreign_steps_are_untouched(run):
    async def main():
        engine, backend = make_engine(
            {"task_done": lambda q: answer(q, "done", 0.95)}, {"stop": {"mode": "guard"}})
        m = msgs(2)
        await engine.decide(m, {"session_id": "s1"})
        h = hook(engine)
        tok = in_check.set(True)
        try:
            assert await h.async_post_call_success_deployment_hook(rd(m), resp(), "completion") is None
        finally:
            in_check.reset(tok)
        assert await h.async_post_call_success_deployment_hook(rd(m, stream=True), resp(), "completion") is None
        assert await h.async_post_call_success_deployment_hook(rd(m), resp(), "text_completion") is None
        assert await h.async_post_call_success_deployment_hook(rd(msgs(1)), resp(), "completion") is None
        assert "task_done" not in backend.calls
        await engine.close()

    run(main())


def test_the_proxy_async_call_type_is_checked(run):
    # the proxy dispatches the deployment hook with CallTypes.acompletion, not "completion"
    async def main():
        engine, backend = make_engine(
            {"task_done": lambda q: answer(q, "done", 0.95)}, {"stop": {"mode": "guard"}})
        m = msgs(2)
        await engine.decide(m, {"session_id": "s1"})
        h = hook(engine)
        async_completion = SimpleNamespace(value="acompletion") # like litellm's CallTypes.acompletion
        assert await h.async_post_call_success_deployment_hook(rd(m), resp(), async_completion) is None
        assert "task_done" in backend.calls # the check ran, not skipped on the call type
        await engine.close()

    run(main())


def test_a_failed_step_still_settles_its_questions(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine(
                {"next_tool": lambda q: answer(q, "read_file", 0.9)},
                {"tool_select": {"mode": "shadow"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            await JahPreCall(engine=engine).async_pre_call_hook(None, None, rd(m, model="jah-auto"), "completion")
            assert (await engine.store.get("s1")).pending
            await JahStepLogger(engine=engine).async_log_failure_event(
                {"data": {"model": "jah-auto", "messages": m, "metadata": {"session_id": "s1"}},
                 "exception": RuntimeError("boom")},
                None, 10.0, 10.5)
            steps = [json.loads(line) for line in Path(tmp, "steps.jsonl").read_text().splitlines() if line.strip()]
            assert len(steps) == 1 and steps[0]["error"] == "boom"
            # like the standalone proxy's error path: the finished questions are logged, unlabeled
            assert any(r["point"] == "tool_select" and r["effect"] is None for r in decision_rows(tmp))
            assert (await engine.store.get("s1")).pending == {}
            await engine.close()

    run(main())


def test_a_failing_check_does_not_starve_the_others(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, backend = make_engine(
                {"task_done": lambda q: answer(q, "done", 0.95),
                 "next_tool": lambda q: answer(q, "read_file", 0.9)},
                {"stop": {"mode": "guard"}, "tool_select": {"mode": "guard"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            await JahPreCall(engine=engine).async_pre_call_hook(None, None, rd(m, model="jah-auto"), "completion")
            assert "tool_select" in (await engine.store.get("s1")).pending
            h = hook(engine)

            async def boom(*a, **k):
                raise RuntimeError("the stop check blew up")

            h._stop_check = boom
            out = await h.async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is None  # the broken check neither redid nor crashed the request
            rows = [r for r in decision_rows(tmp) if r["point"] == "tool_select"]
            assert len(rows) == 1 and rows[0]["effect"] == "accepted"  # the other check settled
            await engine.close()

    run(main())


def test_label_redo_survives_a_failing_tool_ask(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine(
                {"task_done": lambda q: answer(q, "not_done", 0.95),
                 "next_tool": lambda q: answer(q, "read_file", 0.99)},
                {"stop": {"mode": "guard"}, "tool_select": {"mode": "guard"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            await JahPreCall(engine=engine).async_pre_call_hook(None, None, rd(m, model="jah-auto"), "completion")
            sess = await engine.store.get("s1")
            broken = asyncio.get_running_loop().create_future()
            broken.set_exception(RuntimeError("the tool ask blew up"))
            sess.pending["tool_select"] = broken  # the tool guard's future dies with an exception

            async def recall(model, messages, tools, request_data):
                return resp(tool_calls=[tool("read_file")], model="g-strong")

            out = await hook(engine, recall).async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is not None and out.model == "g-strong"  # the stop redo still happened
            rows = [r for r in decision_rows(tmp) if r["point"] == "stop"]
            assert rows and rows[0]["effect"] == "redo" # settle ran: the asks were logged
            assert (await engine.store.get("s1")).pending == {}
            await engine.close()

    run(main())


def test_routing_guard_redoes_a_bad_fast_step(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine(
                {"fast_step_ok": lambda q: answer(q, "redo", 0.95)},
                {"stop": {"mode": "off"}}, log_dir=tmp, mode="guard")
            m = msgs(2)
            res = await engine.decide(m, {"session_id": "s1"})
            assert res.tier == "fast" and res.reason == "guard" # the fast tier writes the step
            calls = []

            async def recall(model, messages, tools, request_data):
                calls.append(model)
                return resp(tool_calls=[tool("bash")], model="g-strong")

            out = await hook(engine, recall).async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is not None and out.model == "g-strong" and calls == ["g-strong"]
            sess = await engine.store.get("s1")
            assert sess.decided[2]["tier"] == "strong" and sess.decided[2]["reason"] == "route_guard"
            rows = [r for r in decision_rows(tmp) if r["point"] == "routing"]
            assert len(rows) == 1
            assert rows[0]["mode"] == "guard" and rows[0]["choice"] == "redo" and rows[0]["effect"] == "redo"
            await engine.close()

    run(main())


def test_routing_guard_accepts_a_sensible_step(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine(
                {"fast_step_ok": lambda q: answer(q, "keep", 0.95)},
                {"stop": {"mode": "off"}}, log_dir=tmp, mode="guard")
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            called = []

            async def recall(*a):
                called.append(1)
                return resp(tool_calls=[tool("bash")], model="g-strong")

            out = await hook(engine, recall).async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is None and not called # the fast reply stands
            rows = [r for r in decision_rows(tmp) if r["point"] == "routing"]
            assert len(rows) == 1 and rows[0]["choice"] == "keep" and rows[0]["effect"] == "accepted"
            await engine.close()

    run(main())


def test_routing_guard_skips_a_final_answer_the_finish_check_redoes(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            # guard mode, stop left at its default (strong): a final answer is re-done by the finish
            # check anyway, so the routing guard must not ask as well
            engine, _ = make_engine({}, None, log_dir=tmp, mode="guard")
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            called = []

            async def recall(model, messages, tools, request_data):
                called.append(model)
                return resp(model="g-strong")

            out = await hook(engine, recall).async_post_call_success_deployment_hook(rd(m), resp(), "completion")
            assert out is not None and out.model == "g-strong" and called == ["g-strong"]
            assert not [r for r in decision_rows(tmp) if r["point"] == "routing"]
            await engine.close()

    run(main())


def test_expand_env_reads_the_reference_at_call_time(monkeypatch):
    from litellm_jah.postcall import _expand_env

    assert _expand_env(None) is None
    assert _expand_env("") is None
    assert _expand_env("a-raw-key") == "a-raw-key"
    monkeypatch.setenv("JAH_RECALL_KEY", "secret")
    assert _expand_env("os.environ/JAH_RECALL_KEY") == "secret"
    assert _expand_env("os.environ/JAH_RECALL_MISSING") is None
    monkeypatch.setenv("JAH_RECALL_EMPTY", "")
    assert _expand_env("os.environ/JAH_RECALL_EMPTY") is None


def test_recall_expands_environment_references(run, monkeypatch):
    import litellm

    async def go():
        seen = {}

        async def fake_acompletion(**kw):
            seen.update(kw)
            return resp(model="g-strong")

        monkeypatch.setenv("JAH_RECALL_KEY", "secret")
        monkeypatch.setenv("JAH_RECALL_BASE", "http://base.example/v1")
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        eng = JAHRouter(JAHCfg(strong="g-strong", strong_api_key="os.environ/JAH_RECALL_KEY",
                               strong_api_base="os.environ/JAH_RECALL_BASE"))
        out = await JahPostCall(engine=eng)._recall_default("g-strong", [{"role": "user", "content": "x"}], [], {})
        assert out is not None
        assert seen["api_key"] == "secret" and seen["api_base"] == "http://base.example/v1"
        assert seen["model"] == "g-strong"
        await eng.close()

    run(go())


def test_other_model_groups_skip_the_checks(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = make_engine({}, {"stop": {"mode": "off"}}, log_dir=tmp)
            m = msgs(2)
            await engine.decide(m, {"session_id": "s1"})
            other = rd(m, model="openai/other", metadata={"session_id": "s1", "model_group": "somebody-else"})
            out = await hook(engine).async_post_call_success_deployment_hook(other, resp(), "completion")
            assert out is None
            sess = await engine.store.get("s1")
            assert sess.step_points == {}  # never settled
            # no model_group stamp: not filtered by the group check (the rec-None guard skips it instead)
            out = await hook(engine).async_post_call_success_deployment_hook(
                rd(m, metadata={"session_id": "s1"}), resp(), "completion")
            assert out is None
            await engine.close()

    run(main())
