"""``scripts/build_hard_weighted::gold_tool``: canonicalization through ``TOOL_ALIASES``.

The miner (``training/mine_hard_cases::select_capped``) and the build script must apply one
per-tool cap to a whole family of harness spellings (bash/shell_run/shell_wait,
question/ask_human/ask_user_question, ...); the build side previously looked up caps by the
raw gold name, so an alias spelling silently fell back to a separate cap (or the built-in
per-alias caps) instead of the family cap.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_hard_weighted.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_hard_weighted", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bhw = _load_module()


def test_gold_tool_canonicalizes_mine_tool_alias():
    assert bhw.gold_tool({"mine": {"tool": "bash"}}) == "shell"
    assert bhw.gold_tool({"mine": {"tool": "shell_wait"}}) == "shell"
    assert bhw.gold_tool({"mine": {"tool": "question"}}) == "ask_user_question"
    assert bhw.gold_tool({"mine": {"tool": "ask_human"}}) == "ask_user_question"
    assert bhw.gold_tool({"mine": {"tool": "web_fetch"}}) == "webfetch"


def test_gold_tool_canonicalizes_target_fallback():
    row = {"mine": {}, "target": {"shell_run": 0.9, "shell": 0.1}}
    assert bhw.gold_tool(row) == "shell"
    row = {"mine": {}, "target": {"web_search": 1.0}}
    assert bhw.gold_tool(row) == "websearch"


def test_gold_tool_non_alias_name_passes_through():
    assert bhw.gold_tool({"mine": {"tool": "edit"}}) == "edit"
    assert bhw.gold_tool({"mine": {"tool": "reply_without_tool"}}) == "reply_without_tool"


def test_family_cap_is_effective_across_spellings():
    # With the v5-style caps file ("shell": 0) the whole shell family is excluded at build
    # time, not just the canonical "shell" spelling.
    caps = dict(bhw.TOOL_CAPS)
    caps.update({"shell": 0, "edit": 5})
    for spelling in ("shell", "bash", "shell_run", "shell_wait"):
        assert caps.get(bhw.gold_tool({"mine": {"tool": spelling}}), 1) == 0
    assert caps.get(bhw.gold_tool({"mine": {"tool": "edit"}}), 1) == 5
