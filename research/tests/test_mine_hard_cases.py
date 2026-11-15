"""``training/mine_hard_cases::select_capped``: per-tool caps, per-session caps, and the default
per-(priority, tool) buckets."""

from __future__ import annotations

from jevharness.training.mine_hard_cases import select_capped


def _row(session: str, tool: str, priority: str = "P0") -> dict:
    return {"meta": {"session": session}, "mine": {"priority": priority, "gold": tool}}


def test_tool_caps_override_default_and_zero_excludes():
    rows = [_row(f"s{i}", "shell") for i in range(10)] + [_row("s90", "reply_without_tool")]
    sel = select_capped(rows, max_per_tool=5, tool_caps={"shell": 0, "reply_without_tool": 1})
    assert [r["mine"]["gold"] for r in sel] == ["reply_without_tool"]  # shell excluded by cap 0


def test_tool_caps_count_across_priorities():
    rows = [_row("a", "edit", "P0"), _row("b", "edit", "P1"), _row("c", "edit", "P1")]
    sel = select_capped(rows, max_per_tool=10, tool_caps={"edit": 2})
    assert len(sel) == 2  # one shared cap across P0+P1, not per-bucket


def test_default_cap_is_per_priority_tool_bucket():
    rows = [_row(f"s{i}", "edit", "P0") for i in range(3)] + [_row("s9", "edit", "P1")]
    sel = select_capped(rows, max_per_tool=2)
    # two P0-edit rows fill the P0 bucket; the P1 bucket is independent
    assert [r["mine"]["priority"] for r in sel] == ["P0", "P0", "P1"]


def test_max_per_session_caps_rows_per_session():
    rows = [_row("s1", "edit") for _ in range(4)] + [_row("s2", "edit")]
    sel = select_capped(rows, max_per_tool=10, max_per_session=2)
    assert len(sel) == 3
    assert [r["meta"]["session"] for r in sel].count("s1") == 2


def test_session_cap_is_per_session_tool_pair():
    rows = [_row("s1", "edit") for _ in range(3)] + [_row("s1", "reply_without_tool") for _ in range(3)]
    sel = select_capped(rows, max_per_tool=10, max_per_session=2)
    # the cap is per (session, tool) pair: s1 keeps 2 edit AND 2 reply (not 2 in total), so a
    # long session with compress errors is not starved by its own reply/edit rows
    assert len(sel) == 4
    tools = [r["mine"]["gold"] for r in sel]
    assert tools.count("edit") == 2
    assert tools.count("reply_without_tool") == 2


def test_sessions_without_meta_are_not_capped():
    rows = [{"mine": {"priority": "P0", "gold": "edit"}} for _ in range(5)]
    assert len(select_capped(rows, max_per_tool=10, max_per_session=2)) == 5


def test_max_total_stops_the_queue():
    rows = [_row(f"s{i}", "edit") for i in range(10)]
    assert len(select_capped(rows, max_per_tool=10, max_total=3)) == 3


def test_input_order_preserved():
    rows = [_row(f"s{i}", "read") for i in range(6)]
    sel = select_capped(rows, max_per_tool=10, max_total=4)
    assert [r["meta"]["session"] for r in sel] == ["s0", "s1", "s2", "s3"]


def test_alias_family_shares_one_cap():
    # question / ask_human / ask_user_question are the same capability under different harness
    # names, so the single "ask_user_question" cap is counted across all three
    rows = [_row("a", "question"), _row("b", "ask_human"),
            _row("c", "ask_user_question"), _row("d", "question")]
    sel = select_capped(rows, max_per_tool=10, tool_caps={"ask_user_question": 2})
    assert len(sel) == 2
    assert sorted(r["mine"]["gold"] for r in sel) == ["ask_human", "question"]


def test_shell_alias_family_is_excluded_by_zero_cap():
    rows = [_row("a", "bash"), _row("b", "shell_run"), _row("c", "shell"), _row("d", "edit")]
    sel = select_capped(rows, max_per_tool=10, tool_caps={"shell": 0, "edit": 5})
    assert [r["mine"]["gold"] for r in sel] == ["edit"]
