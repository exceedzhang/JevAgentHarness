"""``scripts/mine_boundary_pairs::select_boundary_pairs``: structural boundary-pair selection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mine_boundary_pairs import select_boundary_pairs  # noqa: E402


def _row(session: str, step: int, tool: str) -> dict:
    return {
        "id": f"{session}/{step}",
        "state": f"state {session} {step} {tool}",
        "question": "q",
        "options": ["read", "reply_without_tool", "edit"],
        "target": {"read": 1.0 if tool == "read" else 0.0,
                   "reply_without_tool": 1.0 if tool == "reply_without_tool" else 0.0,
                   "edit": 1.0 if tool == "edit" else 0.0},
        "meta": {"session": session, "step": step},
    }


FAM = {"read", "reply_without_tool", "edit"}


def test_adjacent_alternation_pairs():
    rows = [_row("s", 1, "read"), _row("s", 2, "reply_without_tool"), _row("s", 3, "edit")]
    members, pairs = select_boundary_pairs(rows, FAM, max_gap=2)
    assert len(pairs) == 2          # (1,2) and (2,3); the middle row is in both
    assert members == {0, 1, 2}


def test_gap_tolerance():
    rows = [_row("s", 1, "read"), _row("s", 3, "reply_without_tool")]
    assert len(select_boundary_pairs(rows, FAM, max_gap=2)[1]) == 1   # one skipped step ok
    assert len(select_boundary_pairs(rows, FAM, max_gap=1)[1]) == 0   # ...but gap 3 is not


def test_far_steps_do_not_pair():
    rows = [_row("s", 1, "read"), _row("s", 10, "reply_without_tool")]
    assert len(select_boundary_pairs(rows, FAM, max_gap=2)[1]) == 0


def test_same_tool_does_not_pair():
    rows = [_row("s", 1, "read"), _row("s", 2, "read")]
    assert select_boundary_pairs(rows, FAM, max_gap=2) == (set(), [])


def test_sessions_do_not_mix():
    rows = [_row("a", 1, "read"), _row("b", 2, "reply_without_tool")]
    assert select_boundary_pairs(rows, FAM, max_gap=2) == (set(), [])


def test_non_family_tools_do_not_pair():
    rows = [_row("s", 1, "read"), _row("s", 2, "shell")]
    assert len(select_boundary_pairs(rows, FAM, max_gap=2)[1]) == 0
