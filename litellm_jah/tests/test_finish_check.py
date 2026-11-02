"""The finish check (P2): a fast-tier final reply (no tool call) is re-run on the strong tier."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from litellm_jah.config import JAHCfg
from litellm_jah.engine import JAHRouter
from litellm_jah.hook import JahFinishCheck, pass_through
from litellm_jah.logger import JahStepLogger

from conftest import FakeBackend, decision, msgs

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "d", "parameters": {}}}]


def resp(tool_calls=None, model="g-fast", in_t=100, out_t=20, finish=None):
    usage = SimpleNamespace(
        prompt_tokens=in_t, completion_tokens=out_t,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
    msg = SimpleNamespace(tool_calls=tool_calls, content=None if tool_calls else "final answer")
    return SimpleNamespace(usage=usage, model=model,
                           choices=[SimpleNamespace(message=msg,
                                                     finish_reason=finish or ("tool_calls" if tool_calls else "stop"))])


def tool(name="bash", args='{"command": "ls"}'):
    return SimpleNamespace(id="tc1", type="function", function=SimpleNamespace(name=name, arguments=args))


def make_engine(p_hard=0.2, verify_finish=True, strong="g-strong", log_dir=None):
    cfg = JAHCfg(decision={"type": "rules"}, verify_finish=verify_finish, strong=strong, log_dir=log_dir)
    return JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(p_hard)))


def rd(messages, tools=TOOLS, **extra):
    d = {"model": "jah-auto", "messages": messages, "metadata": {"session_id": "s1"}, "temperature": 0.2}
    if tools is not None:
        d["tools"] = tools
    d.update(extra)
    return d


async def routed_step(engine, p_hard=0.2):
    m = msgs(2)
    r = await engine.decide(m, {"session_id": "s1"})
    return m, r


def test_fast_final_reply_is_redone_on_strong(run):
    async def main():
        engine = make_engine()
        seen = []

        async def recall(model, messages, tools, request_data):
            seen.append((model, messages, tools, request_data))
            return resp(tool_calls=[tool()], model="g-strong", in_t=110, out_t=30)

        hook = JahFinishCheck(engine=engine, recall=recall)
        m, r = await routed_step(engine)
        assert r.tier == "fast"
        out = await hook.async_post_call_success_deployment_hook(rd(m), resp(model="g-fast"), "completion")
        assert out is not None
        assert seen and seen[0][0] == "g-strong"
        assert seen[0][1] == m and seen[0][2] == TOOLS
        assert seen[0][3]["temperature"] == 0.2
        assert not ({"model", "messages", "tools", "metadata"} & set(seen[0][3]))
        sess = await engine.store.get("s1")
        rec = sess.decided[2]
        assert rec["tier"] == "strong" and rec["reason"] == "finish_check"
        assert sess.fast_streak == 0 and sess.last_tier == "strong"
        calls = sess.redo_calls[2]
        assert calls[0]["tier"] == "fast" and calls[0]["model"] == "g-fast"
        assert calls[0]["in"] == 100 and calls[0]["out"] == 20 and calls[0]["latency_s"] is None
        assert calls[1]["tier"] == "strong" and calls[1]["model"] == "g-strong"
        assert calls[1]["latency_s"] >= 0.0
        # a retry of the same step is pinned to the strong tier
        again = await engine.decide(m, {"session_id": "s1"})
        assert again.tier == "strong" and again.reason.startswith("repeat:")
        await engine.close()

    run(main())


def test_tool_call_reply_is_not_checked(run):
    async def main():
        engine = make_engine()
        hook = JahFinishCheck(engine=engine, recall=lambda *a: (_ for _ in ()).throw(AssertionError("no recall")))
        m, _ = await routed_step(engine)
        out = await hook.async_post_call_success_deployment_hook(rd(m), resp(tool_calls=[tool()], model="g-fast"),
                                                                 "completion")
        assert out is None
        await engine.close()

    run(main())


def test_strong_tier_step_is_not_checked(run):
    async def main():
        engine = make_engine(p_hard=0.9)
        hook = JahFinishCheck(engine=engine, recall=lambda *a: (_ for _ in ()).throw(AssertionError("no recall")))
        m, r = await routed_step(engine)
        assert r.tier == "strong"
        out = await hook.async_post_call_success_deployment_hook(rd(m), resp(model="g-strong"), "completion")
        assert out is None
        await engine.close()

    run(main())


def test_unrouted_step_is_not_checked(run):
    async def main():
        engine = make_engine()
        hook = JahFinishCheck(engine=engine, recall=lambda *a: (_ for _ in ()).throw(AssertionError("no recall")))
        out = await hook.async_post_call_success_deployment_hook(rd(msgs(2)), resp(model="g-fast"), "completion")
        assert out is None
        await engine.close()

    run(main())


def test_check_disabled(run):
    async def main():
        for kw in ({"verify_finish": False}, {"strong": None}):
            engine = make_engine(**kw)
            hook = JahFinishCheck(engine=engine, recall=lambda *a: (_ for _ in ()).throw(AssertionError("no recall")))
            m, _ = await routed_step(engine)
            out = await hook.async_post_call_success_deployment_hook(rd(m), resp(model="g-fast"), "completion")
            assert out is None
            await engine.close()

    run(main())


def test_request_without_tools_is_not_checked(run):
    async def main():
        engine = make_engine()
        hook = JahFinishCheck(engine=engine, recall=lambda *a: (_ for _ in ()).throw(AssertionError("no recall")))
        m, _ = await routed_step(engine)
        out = await hook.async_post_call_success_deployment_hook(rd(m, tools=None), resp(model="g-fast"),
                                                                 "completion")
        assert out is None
        await engine.close()

    run(main())


def test_stream_and_other_call_types_are_not_checked(run):
    async def main():
        engine = make_engine()
        hook = JahFinishCheck(engine=engine, recall=lambda *a: (_ for _ in ()).throw(AssertionError("no recall")))
        m, _ = await routed_step(engine)
        assert await hook.async_post_call_success_deployment_hook(rd(m, stream=True), resp(model="g-fast"),
                                                                  "completion") is None
        assert await hook.async_post_call_success_deployment_hook(rd(m), resp(model="g-fast"),
                                                                  "text_completion") is None
        await engine.close()

    run(main())


def test_strong_failure_keeps_the_fast_reply(run):
    async def main():
        engine = make_engine()

        async def recall(*a):
            raise RuntimeError("strong tier down")

        hook = JahFinishCheck(engine=engine, recall=recall)
        m, _ = await routed_step(engine)
        out = await hook.async_post_call_success_deployment_hook(rd(m), resp(model="g-fast"), "completion")
        assert out is None
        sess = await engine.store.get("s1")
        assert sess.decided[2]["tier"] == "fast" and "finish_check" not in str(sess.decided[2]["reason"])
        assert 2 not in sess.redo_calls
        await engine.close()

    run(main())


def test_pass_through_drops_only_the_replacement_keys():
    out = pass_through({"model": "jah-auto", "messages": [], "tools": TOOLS, "metadata": {"a": 1},
                        "litellm_metadata": {}, "container": {}, "user_api_key_dict": "x",
                        "temperature": 0.2, "max_tokens": 99, "tool_choice": "auto"})
    assert out == {"temperature": 0.2, "max_tokens": 99, "tool_choice": "auto"}


def test_pass_through_drops_litellm_plumbing_and_credentials():
    # the deployment hook's request_data carries LiteLLM's internals: forwarding the logging object
    # would make the re-run reuse the original request's success log (so the original never fires)
    out = pass_through({"model": "m", "messages": [], "tools": TOOLS,
                        "litellm_logging_obj": object(), "litellm_call_id": "c1",
                        "litellm_trace_id": "t1", "_context_compaction_state": {},
                        "api_key": "sk-tier", "api_base": "http://fast", "api_version": "v1",
                        "client": object(), "shared_session": object(), "proxy_server_request": {},
                        "model_info": {}, "secret_fields": [], "use_litellm_proxy": True,
                        "complexity_router_config": {}, "timeout": 30, "max_retries": 2,
                        "temperature": 0.1, "stream": False})
    assert out == {"timeout": 30, "max_retries": 2, "temperature": 0.1, "stream": False}


def test_step_row_records_both_calls(run):
    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(log_dir=tmp)
            strong = resp(tool_calls=[tool()], model="g-strong", in_t=110, out_t=30)

            async def recall(model, messages, tools, request_data):
                return strong

            hook = JahFinishCheck(engine=engine, recall=recall)
            logger = JahStepLogger(engine=engine)
            m, _ = await routed_step(engine)
            await hook.async_post_call_success_deployment_hook(rd(m), resp(model="g-fast"), "completion")
            await logger.async_log_success_event(
                {"data": {"model": "jah-auto", "messages": m, "metadata": {"session_id": "s1"}},
                 "model": "g-strong"},
                strong, 10.0, 11.0)
            lines = Path(tmp, "steps.jsonl").read_text().splitlines()
            rows = [json.loads(line) for line in lines if line.strip()]
            assert len(rows) == 1
            row = rows[0]
            assert row["tier"] == "strong" and row["reason"] == "finish_check"
            assert [c["tier"] for c in row["calls"]] == ["fast", "strong"]
            assert row["calls"][0]["model"] == "g-fast" and row["calls"][1]["model"] == "g-strong"
            assert row["latency_s"] == 1.0 + row["calls"][1]["latency_s"]
            assert row["n_tool_calls"] == 1 and row["finish"] == "tool_calls"
            assert row["input_tokens"] == 110 and row["output_tokens"] == 30
            await engine.close()

    run(main())
