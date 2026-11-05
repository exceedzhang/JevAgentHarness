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


def test_jev_price_per_m_overrides_the_fallback():
    assert JevBackend(price_per_m=0).price_per_m == 0.0
    assert JevBackend(price_per_m=2).price_per_m == 2.0
    assert JevBackend().price_per_m > 0  # the built-in fallback
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            JevBackend(price_per_m=bad)


def test_make_backend_passes_the_price():
    from litellm_jah.backend import make_backend

    assert make_backend({"type": "jev", "price_per_m": 0}).price_per_m == 0.0


def test_calibration_bucket_boundaries():
    from litellm_jah.backend import _bucket

    assert _bucket(2) == "2" and _bucket(3) == "3-5" and _bucket(5) == "3-5"
    assert _bucket(6) == "6-16" and _bucket(16) == "6-16" and _bucket(17) == "17-52" and _bucket(52) == "17-52"


def test_check_calibration_fails_closed():
    from litellm_jah.backend import _check_calibration

    assert _check_calibration(None) is None
    assert _check_calibration({"global": 0.772}) == {"global": 0.772, "buckets": {}}
    for bad in ([], "x", {}, {"buckets": {}}, {"global": 0}, {"global": -1}, {"global": float("nan")},
                {"global": float("inf")}, {"global": 0.7, "buckets": {"choice:99": 0.5}},
                {"global": 0.7, "buckets": {"noul:2": "warm"}}):
        with pytest.raises(ValueError):
            _check_calibration(bad)


def test_jev_backend_applies_the_card_temperature(run):
    T = 0.7720006824268275  # the card's global temperature; routing/stop are 2-option choice questions

    async def handle(request: web.Request) -> web.Response:
        return web.json_response({"answers": {"next_step": {"probabilities": {"routine": 0.3, "hard": 0.7}}}})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/systemone", handle)
        server = TestServer(app)
        await server.start_server()
        be = JevBackend(provider="local", base_url=str(server.make_url("")), key_env="", retries=0,
                        calibration={"global": T})
        try:
            d = await be.decide("S", QUESTION)
            e = 1.0 / T
            assert d.p["hard"] == pytest.approx(0.7 ** e / (0.7 ** e + 0.3 ** e), abs=1e-9)
            assert sum(d.p.values()) == pytest.approx(1.0) and d.calibrated
        finally:
            await be.close()
            await server.close()
    run(go())


def test_calibration_takes_the_bucket_then_the_global(run):
    probs = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}  # a 4-option question: bucket choice:3-5

    async def handle(request: web.Request) -> web.Response:
        return web.json_response({"answers": {"next_tool": {"probabilities": probs}}})

    async def go():
        app = web.Application()
        app.router.add_post("/v1/systemone", handle)
        server = TestServer(app)
        await server.start_server()
        base = str(server.make_url(""))
        q = Question(id="next_tool", text="t", options=["a", "b", "c", "d"])
        be = JevBackend(provider="local", base_url=base, key_env="", retries=0,
                        calibration={"global": 2.0, "buckets": {"choice:3-5": 0.5}})
        try:
            d = await be.decide("S", q)
            w = [v ** (1.0 / 0.5) for v in probs.values()]  # the bucket temperature, not the global one
            s = sum(w)
            for k, want in zip(q.options, [x / s for x in w]):
                assert d.p[k] == pytest.approx(want, abs=1e-9)
            be2 = JevBackend(provider="local", base_url=base, key_env="", retries=0)
            d2 = await be2.decide("S", q)  # without the block: the raw probabilities, bit-for-bit as before
            assert not d2.calibrated and d2.p == probs
        finally:
            await be.close()
            await be2.close()
            await server.close()
    run(go())


def test_make_backend_passes_the_calibration():
    from litellm_jah.backend import make_backend

    be = make_backend({"type": "jev", "price_per_m": 0, "calibration": {"global": 0.772}})
    assert be.calibration == {"global": 0.772, "buckets": {}}
    assert be.describe()["calibrated"] is True
    with pytest.raises(ValueError):
        make_backend({"type": "jev", "calibration": {"global": 0}})


def test_describe_without_calibration_is_not_calibrated():
    assert JevBackend().describe()["calibrated"] is False
