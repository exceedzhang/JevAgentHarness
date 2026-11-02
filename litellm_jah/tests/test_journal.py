"""The engine's decisions.jsonl: the routing decision point in the standalone proxy's Ask.record() shape."""

from __future__ import annotations

import json

from litellm_jah.config import JAHCfg
from litellm_jah.engine import JAHRouter

from conftest import FakeBackend, decision, msgs


def rows_of(path, name="decisions.jsonl"):
    return [json.loads(ln) for ln in (path / name).read_text().splitlines() if ln.strip()]


def test_decision_rows_are_written_when_asked(tmp_path, run):
    async def go():
        cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path))
        r = JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(0.9)))
        await r.decide(msgs(0))  # first step: forced, not asked -> no row
        await r.decide(msgs(1))  # asked -> row
        await r.decide(msgs(2))  # asked -> row
        rows = rows_of(tmp_path)
        assert len(rows) == 2
        row = rows[0]
        assert row["point"] == "routing" and row["mode"] == "cascade"
        assert row["question"] == "next_step" and row["options"] == ["routine", "hard"] and row["n_options"] == 2
        assert row["p"] == {"routine": 0.1, "hard": 0.9} and row["choice"] == "hard" and row["confidence"] == 0.9
        assert row["tau"] == 0.75 and row["act_option"] == "hard" and row["p_act"] == 0.9
        assert row["acted"] is True and row["effect"] == "hard"
        assert row["label"] is None and row["label_tier"] is None and row["agree"] is None and row["fit_ok"] is False
        assert row["model"] == "fake" and row["rounds"] == 1 and row["error"] is None
        assert row["cost_usd"] == 0.0 and row["latency_s"] >= 0.0
        assert row["model_name"] == "jah-auto" and row["step"] == 1 and row["session"]
        assert "ts" in row
        assert rows[1]["step"] == 2
        await r.close()
    run(go())


def test_routine_rows_do_not_act(tmp_path, run):
    async def go():
        cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path))
        r = JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(0.2)))
        await r.decide(msgs(1))
        row = rows_of(tmp_path)[0]
        # in cascade mode the tier is the decision's effect, so a routine call also "acted" (as in JAH)
        assert row["choice"] == "routine" and row["acted"] is True and row["effect"] == "routine"
        await r.close()
    run(go())


def test_shadow_rows_are_logged_but_do_not_act(tmp_path, run):
    async def go():
        cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path), mode="shadow")
        r = JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(0.9)))
        await r.decide(msgs(1))
        row = rows_of(tmp_path)[0]
        assert row["mode"] == "shadow" and row["acted"] is False and row["effect"] == "shadow"
        await r.close()
    run(go())


def test_off_mode_logs_nothing(tmp_path, run):
    async def go():
        cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path), mode="off")
        r = JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(0.9)))
        await r.decide(msgs(1))
        assert rows_of(tmp_path) == []
        await r.close()
    run(go())


def test_a_backend_error_is_logged_with_error(tmp_path, run):
    async def go():
        def down(n, s, q):
            raise RuntimeError("down")

        cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path), on_error="strong")
        r = JAHRouter(cfg, backend=FakeBackend(down))
        res = await r.decide(msgs(1))
        assert res.tier == "strong" and res.reason == "decision_error"
        row = rows_of(tmp_path)[0]
        assert row["error"] and "RuntimeError" in row["error"]
        assert row["p"] == {} and row["choice"] is None and row["p_act"] is None and row["acted"] is False
        assert row["effect"] == "decision_error"
        await r.close()
    run(go())


def test_a_retry_does_not_log_twice(tmp_path, run):
    async def go():
        cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path))
        r = JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(0.2)))
        m = msgs(1)
        await r.decide(m)
        await r.decide(m)  # retry of the same step: cached, no new decision call
        assert len(rows_of(tmp_path)) == 1
        await r.close()
    run(go())


def test_no_logging_without_log_dir(run, tmp_path):
    async def go():
        r = JAHRouter(JAHCfg(decision={"type": "rules"}), backend=FakeBackend(lambda n, s, q: decision(0.9)))
        await r.decide(msgs(1))
        assert r.journal is None and not (tmp_path / "decisions.jsonl").exists()
        await r.close()
    run(go())
