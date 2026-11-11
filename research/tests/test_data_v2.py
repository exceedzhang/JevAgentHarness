"""``training/data_v2``: production ``decisions.jsonl`` rows -> GRPO/SFT examples (state + question +
options must render exactly as served; skip reasons must be visible in the summary)."""

from __future__ import annotations

import json

from jevharness.training.data import render
from jevharness.training.data_v2 import from_production


def _row(point, label=None, fit_ok=True, state="S", opts=None, p=None, ts=1.0,
         session="s1", step=1, error=None, **extra):
    r = {"point": point, "session": session, "step": step, "ts": ts, "label": label,
         "fit_ok": fit_ok, "options": opts or (["done", "not_done"] if point == "stop" else
                                               ["bash", "read", "reply_without_tool"]),
         "p": p or {}, "state": state, "question": "task_done" if point == "stop" else "next_tool",
         "choice": None, "model": "m", "calibrated": True, "truncated": False, "error": error}
    r.update(extra)
    return r


def test_labelled_rows_become_onehot_examples():
    rows = [_row("tool_select", label="bash", p={"bash": 0.2, "read": 0.5, "reply_without_tool": 0.3})]
    ex, summary = from_production(rows)
    assert len(ex) == 1 and summary["n_out"] == 1 and summary["skipped"] == {}
    e = ex[0]
    assert (e["id"], e["source"], e["dp"], e["kind"]) == ("prod:s1/1/tool_select", "production", "tool_select", "choice")
    assert e["target"] == {"bash": 1.0, "read": 0.0, "reply_without_tool": 0.0}
    prompt, codes = render(e)
    assert "STATE:\nS" in prompt and "QUESTION: next_tool" in prompt and "A. bash" in prompt
    # the target over the presented codes sums to one
    s = sum(e["target"].values())
    assert abs(s - 1.0) < 1e-9
    assert ex[0]["meta"]["label_tier"] is None  # absent in the synthetic row, stays None


def test_routing_stop_and_invalid_rows_are_skipped_with_reasons():
    rows = [
        _row("routing", p={"routine": 0.4, "hard": 0.6}),                       # no label source
        _row("stop", state=None),                                               # state missing
        _row("stop", label="not_done", fit_ok=False),                           # label produced by the model
        _row("tool_select", label="bash", error="boom"),                        # failed decision
        _row("tool_select", label="grep"),                                     # tool renamed away from options
        _row("tool_select", label="bash", session="s1", step=1),                # first one already used below
        _row("tool_select", label="bash", session="s1", step=1),                # duplicate (session, step, point)
    ]
    rows.insert(0, _row("stop", label="done", p={"done": 0.7, "not_done": 0.3}))
    ex, summary = from_production(rows)
    ids = {e["id"] for e in ex}
    assert ids == {"prod:s1/1/stop", "prod:s1/1/tool_select"}
    assert summary["n_in"] == len(rows)
    for k in ("point_unlabelled", "no_state", "unlabelled_or_not_fit_ok", "error_row",
              "label_not_in_options", "duplicate"):
        assert summary["skipped"].get(k) == 1, (k, summary["skipped"])
    assert summary["n_out"] == 2


def test_routing_can_be_included_as_a_soft_target():
    rows = [_row("routing", p={"routine": 0.4, "hard": 0.6})]
    ex, summary = from_production(rows, include_routing=True)
    assert len(ex) == 1
    e = ex[0]
    assert e["dp"] == "routing" and e["meta"]["label_source"] == "model_soft"
    assert abs(sum(e["target"].values()) - 1.0) < 1e-9


def test_same_session_step_different_states_are_distinct_samples():
    # clients reuse one session id across tasks and the step number restarts: same (session, step)
    # with different states are two real steps, only an identical state is a retry
    rows = [
        _row("tool_select", label="bash", state="TASK: A  STEPS SO FAR: 1"),
        _row("tool_select", label="read", state="TASK: B  STEPS SO FAR: 1"),
        _row("tool_select", label="bash", state="TASK: A  STEPS SO FAR: 1"),  # true retry of the first
    ]
    ex, summary = from_production(rows)
    assert len(ex) == 2
    assert summary["skipped"] == {"duplicate": 1}


def test_build_writes_split_and_manifest(tmp_path):
    import tempfile

    from jevharness.training.data_v2 import build

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/decisions.jsonl"
        with open(path, "w") as f:
            for s in range(6):
                f.write(json.dumps(_row("tool_select", label="bash", session=f"s{s%3}", step=s, ts=s + 1.0)) + "\n")
        m = build(path, dev_frac=0.34, out_dir=d)
        assert m["n_out"] == 6
        import os

        for fn in ("train.jsonl", "dev.jsonl", "MANIFEST.json"):
            assert os.path.exists(f"{d}/{fn}"), fn
        train = [json.loads(l) for l in open(f"{d}/train.jsonl")]
        dev = [json.loads(l) for l in open(f"{d}/dev.jsonl")]
        assert len(train) + len(dev) == 6
        # task-split: a session goes to exactly one side
        st = {e["meta"]["session"] for e in train}
        sd = {e["meta"]["session"] for e in dev}
        assert not (st & sd)
