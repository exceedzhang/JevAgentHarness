"""Tests for the 27B validity judge module (``training.judge``) -- pure logic only, no network."""
from jevharness.training import judge


def test_clip_state_short_passthrough():
    assert judge.clip_state("abc") == "abc"


def test_clip_state_truncates_middle():
    s = "A" * 1000 + "MIDDLE" + "B" * 10000
    out = judge.clip_state(s, head=10, tail=20)
    assert out.startswith("A" * 10)
    assert out.endswith("B" * 20)
    assert "middle elided" in out


def test_describe_choice():
    assert "FINAL TEXT REPLY" in judge.describe_choice("reply_without_tool")
    assert "calling the tool/action: read" in judge.describe_choice("read")


def test_build_prompt_has_validity_framing():
    case = {"state": "situation text", "opts": ["read", "reply_without_tool"], "pred": "read"}
    p = judge.build_prompt(case)
    assert "VALID, REASONABLE" in p
    assert "Do NOT judge which action is better" in p
    assert "situation text" in p
    assert "The agent CHOSE: calling the tool/action: read" in p


def test_verdict_parsing():
    assert judge.verdict("VALID: fine") == "VALID"
    assert judge.verdict("invalid, wrong file") == "INVALID"
    assert judge.verdict("UNCERTAIN -- unclear") == "UNCERTAIN"
    assert judge.verdict("no verdict here") == "PARSE_ERR"


def test_summarize():
    results = [
        {"verdict": "VALID", "direction": "tool_to_reply"},
        {"verdict": "INVALID", "direction": "tool_to_reply"},
        {"verdict": "VALID", "direction": "reply_to_tool"},
    ]
    s = judge.summarize(results)
    assert s["all"]["n"] == 3
    assert s["tool_to_reply"]["VALID"] == 0.5
    assert s["reply_to_tool"]["VALID"] == 1.0