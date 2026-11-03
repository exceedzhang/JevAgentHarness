"""Acceptance: the standalone proxy's `jah report` / `jah fit` read the plugin's logs.

Requires the main package (``pip install -e ..`` from the repo root); skipped otherwise.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from litellm_jah.config import JAHCfg
from litellm_jah.engine import JAHRouter
from litellm_jah.logger import JahStepLogger

from conftest import FakeBackend, decision, msgs

jevagentharness = pytest.importorskip("jevagentharness")
from jevagentharness.fit import describe, fit, read_decisions  # noqa: E402
from jevagentharness.report import decisions_table, read_steps, summarize, table  # noqa: E402


def _resp(s: int):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100 + s, completion_tokens=10,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=0)),
        model="gemini/fake-argon-9000",
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[{"id": "t"}]), finish_reason="tool_calls")],
    )


def test_report_and_fit_on_plugin_logs(tmp_path, run):
    async def go():
        cfg = JAHCfg(decision={"type": "rules"}, log_dir=str(tmp_path))
        engine = JAHRouter(cfg, backend=FakeBackend(lambda n, s, q: decision(0.9)))
        logger = JahStepLogger(engine=engine)
        t0 = time.time()
        for s in range(3):  # step 0: first step (strong, not asked); steps 1-2: asked, hard
            m = msgs(s)
            await engine.decide(m, {"session_id": "s1"})
            kwargs = {"data": {"model": "jah-auto", "messages": m, "metadata": {"session_id": "s1"}},
                      "model": "gemini/fake-argon-9000"}
            await logger.async_log_success_event(kwargs, _resp(s), t0, t0 + 1.0)
        await engine.close()
    run(go())

    # `jah report`
    steps = read_steps(tmp_path)
    assert len(steps) == 3
    summary = summarize(steps)
    assert "jah-auto" in summary
    g = summary["jah-auto"]
    assert g["sessions"] == 1 and g["steps"] == 3
    assert g["tier_share"]["strong"] == 1.0
    assert g["reasons"].get("hard", 0) == 2 and g["reasons"].get("first_steps", 0) == 1
    assert g["unpriced_calls"] == 3  # the fake model is not in the cost map
    rendered = table(summary)
    assert "jah-auto" in rendered

    # the decision calls, as `jah report` prints them
    recs = read_decisions(tmp_path)
    described = describe(recs)
    assert "routing:cascade" in described
    assert described["routing:cascade"]["calls"] == 2
    assert described["routing:cascade"]["errors"] == 0
    assert described["routing:cascade"]["acted"] == 2
    assert "routing:cascade" in decisions_table(described)

    # `jah fit`
    res = fit(recs)
    assert res["n_records"] == 2 and res["model_names"] == ["jah-auto"]
    assert res["points"]["tool_select"]["n_labelled"] == 0
    assert "routing:cascade" in res["describe"]
    from jevagentharness.fit import table as fit_table
    txt = fit_table(res)
    assert "routing:cascade" in txt and "tool_select" in txt
