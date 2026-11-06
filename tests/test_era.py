"""Era filters (--since/--decision-model/--calibrated/--truncated) for jah report and jah fit."""

from __future__ import annotations

import json
from datetime import datetime

from jevagentharness import era
from jevagentharness.cli import main


def test_parse_since_formats():
    t = era.parse_since("2026-11-05 17:01")
    assert t == datetime(2026, 11, 5, 17, 1).timestamp()
    assert era.parse_since("2026-11-05") == datetime(2026, 11, 5).timestamp()
    assert era.parse_since("1762000000") == 1762000000.0
    for bad in ("yesterday", "17:01", ""):
        try:
            era.parse_since(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_parse_bool():
    assert era.parse_bool("true") and era.parse_bool("1") and era.parse_bool("YES")
    assert not era.parse_bool("false") and not era.parse_bool("off")
    try:
        era.parse_bool("maybe")
    except ValueError:
        pass
    else:
        raise AssertionError


def drow(ts: float, model: str, **kw: object) -> dict:
    return {"ts": ts, "point": "routing", "model": model, **kw}


def test_era_keep():
    raw = drow(100.0, "AgentDecision-4B")                       # pre-era fields: absent = raw, uncut
    cal = drow(200.0, "AgentDecision-4B", calibrated=True, truncated=False)
    prev = drow(300.0, "decision-model-preview", calibrated=None, truncated=None)
    e = era.Era(since=150.0)
    assert [r for r in (raw, cal, prev) if e.keep(r)] == [cal, prev]
    assert era.Era(model="4b").keep(cal) and not era.Era(model="4b").keep(prev)
    assert era.Era(calibrated=True).keep(cal)
    assert not era.Era(calibrated=True).keep(raw)               # absent counts as the raw era
    assert era.Era(calibrated=False).keep(raw) and era.Era(calibrated=False).keep(prev)
    assert era.Era(truncated=True).keep(drow(1.0, "x", truncated=True))
    assert not era.Era(truncated=True).keep(drow(1.0, "x", error="boom", truncated=None))
    assert e.note() == "since " + datetime.fromtimestamp(150.0).strftime("%Y-%m-%d %H:%M")
    assert not era.Era().note()


def test_apply_steps_keeps_cache_rows_of_surviving_sessions():
    steps = [
        {"ts": 100.0, "session": "a", "kind": None, "decision": {"model": "AgentDecision-4B"}},
        {"ts": 100.0, "session": "a", "kind": "cache", "cost_usd": 1},
        {"ts": 400.0, "session": "b", "kind": None, "decision": {"model": "decision-model-preview"}},
        {"ts": 400.0, "session": "b", "kind": "cache", "cost_usd": 2},
    ]
    kept = era.apply_steps(steps, era.Era(model="4B"))
    assert [r.get("kind") for r in kept] == [None, "cache"] and kept[1]["session"] == "a"
    assert era.apply_steps(steps, era.Era()) == steps           # no filter: identity
    assert era.apply_steps(steps, era.Era(since=200.0)) == steps[2:]


def _write_logs(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    steps = [
        {"ts": 100.0, "session": "p", "model_name": "jah-auto", "step": 0, "tier": "fast", "reason": "routine",
         "cost_usd": 0.1, "latency_s": 1, "decision": {"model": "AgentDecision-4B", "latency_s": 0.2}},
        {"ts": 200.0, "session": "q", "model_name": "jah-auto", "step": 0, "tier": "strong", "reason": "hard",
         "cost_usd": 0.5, "latency_s": 5, "p_hard": 0.9, "decision": {"model": "decision-model-preview"}},
        {"ts": 200.0, "session": "q", "kind": "cache", "model_name": "jah-auto", "cost_usd": 0.3},
    ]
    decs = [drow(100.0, "AgentDecision-4B", mode="cascade", p_act=0.4, tau=0.8, latency_s=0.2, cost_usd=0.0),
            drow(200.0, "decision-model-preview", mode="cascade", p_act=0.9, tau=0.8, latency_s=0.4, cost_usd=0.1)]
    (d / "steps.jsonl").write_text("\n".join(json.dumps(x) for x in steps))
    (d / "decisions.jsonl").write_text("\n".join(json.dumps(x) for x in decs))
    return d


def test_report_and_fit_accept_era_flags(tmp_path, capsys):
    d = _write_logs(tmp_path)
    assert main(["report", str(d), "--decision-model", "4b"]) == 0
    cap = capsys.readouterr()
    assert "1.0 |" in cap.out and "0.1000" in cap.out          # only the 4B session survives
    assert "# era filter: decision model ~'4b'" in cap.err
    assert "preview" not in cap.out

    capsys.readouterr()
    main(["fit", str(d), "--calibrated", "false"])
    assert "too few labelled" in capsys.readouterr().out        # the table still prints, filtered


def test_fit_snippet_records_the_era(tmp_path, capsys):
    d = _write_logs(tmp_path)
    out = tmp_path / "th.yaml"
    main(["fit", str(d), "--since", "0", "--out", str(out)])
    text = out.read_text()
    assert "# era filter: since " in text.splitlines()[1]
    assert "jah fit" in text.splitlines()[0]
