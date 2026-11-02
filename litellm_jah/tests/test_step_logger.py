"""The steps.jsonl logger: JAH step rows from LiteLLM's success/failure events (duck-typed, no litellm)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from litellm_jah.config import JAHCfg
from litellm_jah.context import current_step
from litellm_jah.engine import JAHRouter
from litellm_jah.logger import JahStepLogger
from litellm_jah.precall import JahPreCall

from conftest import FakeBackend, decision, msgs


def fake_response(model="gemini/gemini-4-argon", in_t=120, out_t=30, cached=40, tool_calls=1, finish="tool_calls"):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=in_t, completion_tokens=out_t,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=0)),
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[{"id": "t"}] * tool_calls),
                                 finish_reason=finish)],
    )


def fake_kwargs(messages, model="jah-auto", **metadata):
    return {"data": {"model": model, "messages": messages, "metadata": metadata}, "model": "gemini/gemini-4-argon"}


def rows_of(path):
    return [json.loads(ln) for ln in (path / "steps.jsonl").read_text().splitlines() if ln.strip()]


async def routed_step(tmp_path, p=0.9):
    cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path))
    engine = JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(p)))
    logger = JahStepLogger(engine=engine)
    m = msgs(1)
    md = {"session_id": "s1"}
    res = await engine.decide(m, md)
    return engine, logger, m, md, res


def test_step_row_for_a_routed_step(tmp_path, run):
    async def go():
        engine, logger, m, md, res = await routed_step(tmp_path, p=0.9)
        t0 = time.time() - 1.25
        await logger.async_log_success_event(fake_kwargs(m, **md), fake_response(), t0, time.time())
        rows = rows_of(tmp_path)
        assert len(rows) == 1
        r = rows[0]
        assert r["session"] == "s1" and r["step"] == 1 and r["model_name"] == "jah-auto"
        assert r["tier"] == "strong" and r["reason"] == "hard" and r["p_hard"] == 0.9
        assert r["input_tokens"] == 120 and r["output_tokens"] == 30 and r["cached_tokens"] == 40
        assert r["n_tool_calls"] == 1 and r["finish"] == "tool_calls"
        assert r["latency_s"] == pytest.approx(1.25, abs=0.01) and r["calls"][0]["tier"] == "strong"
        assert r["calls"][0]["priced"] is False  # no litellm in the test env: logged as unpriced
        assert r["cost_usd"] == r["llm_cost_usd"] == 0.0 and r["decision_cost_usd"] == 0.0
        assert "decision" in r and "ts" in r
        await engine.close()
    run(go())


def test_failure_row_carries_the_error(tmp_path, run):
    async def go():
        engine, logger, m, md, _ = await routed_step(tmp_path)
        kwargs = fake_kwargs(m, **md)
        kwargs["exception"] = RuntimeError("upstream 500")
        await logger.async_log_failure_event(kwargs, None, time.time() - 0.5, time.time())
        rows = rows_of(tmp_path)
        assert len(rows) == 1
        assert "upstream 500" in rows[0]["error"]
        assert rows[0]["input_tokens"] == 0 and rows[0]["output_tokens"] == 0
        await engine.close()
    run(go())


def test_unrouted_steps_are_not_logged(tmp_path, run):
    async def go():
        engine, logger, m, md, _ = await routed_step(tmp_path)
        current_step.set(None) # a different request's context: this plugin never routed it
        other = fake_kwargs(msgs(1, task="a different task"), model="jah-auto") # a different session
        await logger.async_log_success_event(other, fake_response(), time.time() - 0.1, time.time())
        assert rows_of(tmp_path) == []
        await engine.close()
    run(go())


def test_missing_data_is_ignored(tmp_path, run):
    async def go():
        engine, logger, m, md, _ = await routed_step(tmp_path)
        current_step.set(None) # a request the plugin never routed
        await logger.async_log_success_event({}, None, time.time(), time.time())
        await logger.async_log_success_event({"data": {}}, None, time.time(), time.time())
        assert rows_of(tmp_path) == []
        await engine.close()
    run(go())


def test_logging_never_raises_on_a_broken_event(tmp_path, run):
    async def go():
        engine, logger, m, md, _ = await routed_step(tmp_path)
        current_step.set(None) # a request the plugin never routed (unknown session): no crash, no row
        await logger.async_log_success_event({"data": {"messages": [{"role": "user", "content": "x"}]}},
                                             object(), time.time(), time.time())
        assert rows_of(tmp_path) == []
        await engine.close()
    run(go())


def test_success_event_with_datetime_bounds_measures_latency(tmp_path, run):
    # litellm passes datetime.datetime start/end (a timedelta apart), not epoch floats
    async def go():
        from datetime import datetime, timedelta
        engine, logger, m, md, _ = await routed_step(tmp_path)
        t0 = datetime.now()
        await logger.async_log_success_event(fake_kwargs(m, **md), fake_response(), t0, t0 + timedelta(seconds=1.5))
        rows = rows_of(tmp_path)
        assert len(rows) == 1
        assert rows[0]["latency_s"] == 1.5 and rows[0]["calls"][0]["latency_s"] == 1.5
        await engine.close()
    run(go())


def test_redo_row_sums_each_calls_own_latency(tmp_path, run):
    # a redone step: the first call's entry carries no latency (the post-call hook has no event times);
    # the logger fills it with the event span, which on litellm 1.106.x covers the first call only (the
    # deployment hook runs after end_time) - the re-run must be counted exactly once
    async def go():
        engine, logger, m, md, _ = await routed_step(tmp_path)
        sess = await engine.store.get("s1")
        sess.redo_calls[1] = [
            {"tier": "fast", "model": "fast-model", "in": 100, "out": 20, "latency_s": None},
            {"tier": "strong", "model": "strong-model", "in": 100, "out": 30, "latency_s": 2.257},
        ]
        t0 = time.time() - 1.149 # the event span: the first call only
        await logger.async_log_success_event(fake_kwargs(m, **md), fake_response(), t0, time.time())
        rows = rows_of(tmp_path)
        assert len(rows) == 1
        r = rows[0]
        assert r["calls"][0]["latency_s"] == 1.149 # first call = the event span
        assert r["calls"][1]["latency_s"] == 2.257 # the re-run = its own measured time
        assert r["latency_s"] == pytest.approx(3.406, abs=0.01) # span + re-run, counted once
        await engine.close()
    run(go())


def test_success_event_in_litellm_shape_uses_the_routed_identity(tmp_path, run):
    # the real litellm event kwargs carry no "data" and no request metadata: messages sit at the top
    # level and the session id comes from the identity ``decide`` left in the request context
    async def go():
        engine, logger, m, md, _ = await routed_step(tmp_path)
        event = {"messages": m, "model": "Qwen/Qwen3.6-27B-FP8", "custom_llm_provider": "openai",
                 "litellm_call_id": "c1"} # no "data", no "metadata"
        await logger.async_log_success_event(event, fake_response(model="Qwen/Qwen3.6-27B-FP8"),
                                             time.time() - 0.5, time.time())
        rows = rows_of(tmp_path)
        assert len(rows) == 1
        assert rows[0]["session"] == "s1" and rows[0]["step"] == 1 # matched by context, not re-derived
        assert rows[0]["model_name"] == "jah-auto" and rows[0]["calls"][0]["model"] == "Qwen/Qwen3.6-27B-FP8"
        await engine.close()
    run(go())


def test_other_model_groups_are_not_logged(tmp_path, run):
    async def go():
        engine, logger, m, md, _ = await routed_step(tmp_path)
        t0 = time.time() - 1.0
        await logger.async_log_success_event(
            fake_kwargs(m, model="somebody-else", **md), fake_response(), t0, time.time())
        assert rows_of(tmp_path) == []
        await engine.close()
    run(go())


def test_streamed_steps_settle_their_questions(tmp_path, run):
    async def go():
        cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path),
                     decisions={"tool_select": {"mode": "guard"}})
        engine = JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(0.9)))
        logger = JahStepLogger(engine=engine)
        pre = JahPreCall(engine=engine)
        m = msgs(1)
        md = {"session_id": "s1"}
        req = {"model": "jah-auto", "messages": m, "metadata": md, "stream": True,
               "tools": [{"type": "function", "function": {"name": "bash", "description": "d", "parameters": {}}}]}
        await pre.async_pre_call_hook(None, None, req, "completion")  # launches the (non-blocking) tool ask
        kw = {"messages": m, "model": "jah-auto", "metadata": md, "stream": True}  # real stream-event shape
        t0 = time.time() - 1.0
        await logger.async_log_stream_event(kw, fake_response(), t0, time.time())
        sess = await engine.store.get("s1")
        assert sess.pending == {}  # settled, not left for the next request
        assert sess.step_points[1]["n_asks"] == 1
        rows = [json.loads(ln) for ln in (tmp_path / "decisions.jsonl").read_text().splitlines() if ln.strip()]
        assert any(r["point"] == "tool_select" for r in rows)
        assert len(rows_of(tmp_path)) == 1  # the step row is written by the stream event too
        await engine.close()
    run(go())


def test_streamed_steps_settle_their_questions_legacy_data_shape(tmp_path, run):
    async def go():
        cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path),
                     decisions={"tool_select": {"mode": "guard"}})
        engine = JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(0.9)))
        logger = JahStepLogger(engine=engine)
        pre = JahPreCall(engine=engine)
        m = msgs(1)
        md = {"session_id": "s1"}
        req = {"model": "jah-auto", "messages": m, "metadata": md, "stream": True,
               "tools": [{"type": "function", "function": {"name": "bash", "description": "d", "parameters": {}}}]}
        await pre.async_pre_call_hook(None, None, req, "completion")
        kw = fake_kwargs(m, **md)
        kw["data"]["stream"] = True
        t0 = time.time() - 1.0
        await logger.async_log_success_event(kw, fake_response(), t0, time.time())
        sess = await engine.store.get("s1")
        assert sess.pending == {}
        assert sess.step_points[1]["n_asks"] == 1
        await engine.close()
    run(go())
