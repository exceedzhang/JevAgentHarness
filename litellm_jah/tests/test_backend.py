"""The decision backends: rules heuristics and the Jev wire format against a local fake server."""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from litellm_jah.backend import JevBackend, Question, RulesBackend, StepView
from litellm_jah.state import QUESTION


def test_rules_backend_judgement():
    assert RulesBackend().judge(StepView()) == (0.1, "no_signal")
    assert RulesBackend().judge(StepView(last_action="bash(x)", prev_action="bash(x)")) == (0.9, "repeated_action")
    assert RulesBackend().judge(StepView(last_output="FAILED: 2 tests")) == (0.9, "error_signal")
    p, why = RulesBackend().judge(StepView(last_output="Traceback (most recent call last):"))
    assert (p, why) == (0.9, "error_signal")


def test_rules_backend_answers_routing_and_refuses_others(run):
    async def go():
        be = RulesBackend()
        d = await be.decide("S", QUESTION, StepView(last_output="all good"))
        assert d.p == {"hard": 0.1, "routine": 0.9} and d.choice == "routine"
        try:
            await be.decide("S", Question(id="next_tool", text="t", options=["a"]), StepView())
            raise AssertionError("the rules backend must refuse next_tool")
        except RuntimeError as e:
            assert "next_tool" in str(e)
    run(go())


def test_question_wire_format():
    w = QUESTION.wire()
    assert w["type"] == "choice" and w["instructions"].startswith("You are watching an AI agent")
    assert set(w["criteria"]) == {"routine", "hard"} and None not in w["criteria"].values()


def test_jev_backend_retries_a_499_and_not_a_400(run):
    calls: list[int] = []
    statuses = [499, 200, 400]

    async def handle(request: web.Request) -> web.Response:
        body = await request.json()
        assert body["questions"]["next_step"]["type"] == "choice" and body["state"] == "S"
        status = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(status)
        if status != 200:
            return web.json_response({"error": {"message": "Unable to complete this request."}}, status=status)
        return web.json_response({"model": "typesafe/jev-test", "usage": {"input_tokens": 2000, "cost": 8.4e-05},
                                  "answers": {"next_step": {"probabilities": {"routine": 0.3, "hard": 0.7}}}})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/systemone", handle)
        server = TestServer(app)
        await server.start_server()
        be = JevBackend(provider="local", base_url=str(server.make_url("")), key_env="", retries=2)
        try:
            d = await be.decide("S", QUESTION)
            assert d.p == {"routine": 0.3, "hard": 0.7} and d.cost_usd == 8.4e-05 and d.model == "typesafe/jev-test"
            assert calls == [499, 200]  # the 499 was retried once
            try:
                await be.decide("S", QUESTION)
                raise AssertionError("a 400 must raise")
            except RuntimeError as e:
                assert "HTTP 400" in str(e)
        finally:
            await be.close()
            await server.close()
    run(go())


def test_jev_backend_normalises_an_unnormalised_distribution(run):
    async def handle(request: web.Request) -> web.Response:
        return web.json_response({"answers": {"next_step": {"probabilities": {"routine": 0.4, "hard": 0.2}}}})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/systemone", handle)
        server = TestServer(app)
        await server.start_server()
        be = JevBackend(provider="local", base_url=str(server.make_url("")), key_env="", retries=0)
        try:
            d = await be.decide("S", QUESTION)
            assert d.p["routine"] == pytest.approx(2 / 3) and d.p["hard"] == pytest.approx(1 / 3)
            assert d.cost_usd == 0  # no usage reported, no tokens -> free fallback price * 0
        finally:
            await be.close()
            await server.close()
    run(go())


def test_jev_backend_rejects_an_empty_distribution(run):
    async def handle(request: web.Request) -> web.Response:
        return web.json_response({"answers": {"next_step": {"probabilities": {}}}})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/systemone", handle)
        server = TestServer(app)
        await server.start_server()
        be = JevBackend(provider="local", base_url=str(server.make_url("")), key_env="", retries=0)
        try:
            try:
                await be.decide("S", QUESTION)
                raise AssertionError("an empty distribution must raise")
            except RuntimeError as e:
                assert "no distribution" in str(e)
        finally:
            await be.close()
            await server.close()
    run(go())
