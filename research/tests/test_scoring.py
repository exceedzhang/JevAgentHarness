"""Tests for the asymmetric decision-validity scoring core (``training.scoring``)."""
import pytest

from jevharness.training import scoring as S


def test_option_name_strips_description():
    assert S.option_name("read") == "read"
    assert S.option_name("read -- show a file") == "read"
    assert S.option_name("  bash  ") == "bash"


def test_is_reply():
    assert S.is_reply("reply_without_tool")
    assert S.is_reply("reply_without_tool -- say done")
    assert not S.is_reply("read")
    assert not S.is_reply("question")


def test_quadrant_all_cases():
    R, T, T2 = "reply_without_tool", "read", "grep"
    assert S.quadrant(T, T) == "match"
    assert S.quadrant(R, T) == "reply_to_tool"   # gold reply, called a tool
    assert S.quadrant(T, R) == "tool_to_reply"   # gold tool, replied early
    assert S.quadrant(T, T2) == "other"          # both tools, different


def test_asymmetric_validity_scores():
    R, T, T2 = "reply_without_tool", "read", "grep"
    assert S.asymmetric_validity(T, T) == 1.0          # match
    assert S.asymmetric_validity(R, T) == 0.0          # reply->tool: not penalized
    assert S.asymmetric_validity(T, R) == -1.0         # tool->reply: heavy penalty
    assert S.asymmetric_validity(T, T2) == 0.0         # other wrong tool: neutral


def test_readonly_bonus_off_by_default_and_on_when_enabled():
    R, RO, WRITE = "reply_without_tool", "grep", "write"
    # off by default: extra tool stays 0.0 even if read-only
    assert S.asymmetric_validity(R, RO, readonly_bonus=False) == 0.0
    # on: read-only extra tool earns partial credit; a non-read-only tool still 0.0
    assert S.asymmetric_validity(R, RO, readonly_bonus=True) == S.READONLY_BONUS
    assert S.asymmetric_validity(R, WRITE, readonly_bonus=True) == 0.0


def test_acceptable_rule():
    R, T, T2 = "reply_without_tool", "read", "grep"
    assert S.acceptable(T, T) is True            # match
    assert S.acceptable(R, T) is True             # reply->tool forgiven
    assert S.acceptable(T, R) is False            # tool->reply penalized
    assert S.acceptable(T, T2) is False           # other wrong tool


def test_aggregation_reproduces_v8_rate():
    """(475 match + 51 reply->tool) / 842 == 62.5%, the v8 asym_valid headline (§13.12)."""
    R, T, T2 = "reply_without_tool", "read", "grep"
    cases = [(T, T)] * 475 + [(R, T)] * 51 + [(T, R)] * 142 + [(T, T2)] * 174
    n = len(cases)
    assert n == 842
    rate = sum(S.acceptable(g, p) for g, p in cases) / n
    assert abs(rate - 0.62470) < 1e-3
    # sanity: strict label-match over the same set is 56.4%
    acc = sum(g == p for g, p in cases) / n
    assert abs(acc - 0.56413) < 1e-3


def test_eval_wires_asym_valid_and_quadrants():
    """Phase 1: `eval_decisions.evaluate` must expose `asym_valid` + the four `quad_*` shares in
    `overall`, and `asym_valid` must equal (match + reply_to_tool)/n, sitting between `acc`
    (label-match) and the judge's all-acceptable bound."""
    from jevharness.training.eval_decisions import evaluate

    class _FakeEngine:
        # one-hot at index 0 => pred = options[0]; gold is controlled per-example via `target`
        def _probs(self, items):
            return [[1.0] + [0.0] * (len(opts) - 1) for (_s, _q, opts, _k) in items]

    exs = [
        # match: gold=read (idx0), pred=read (idx0) -> label-correct
        {"state": "s", "question": "q", "options": ["read", "grep"], "target": {"read": 1.0}, "kind": "choice", "dp": "tool_select"},
        # reply_to_tool: gold=reply_without_tool (idx1), pred=read (idx0) -> acceptable
        {"state": "s", "question": "q", "options": ["read", "reply_without_tool"], "target": {"reply_without_tool": 1.0}, "kind": "choice", "dp": "tool_select"},
        # tool_to_reply: gold=read (idx1), pred=reply_without_tool (idx0) -> NOT acceptable (抢答)
        {"state": "s", "question": "q", "options": ["reply_without_tool", "read"], "target": {"read": 1.0}, "kind": "choice", "dp": "tool_select"},
        # other: gold=read (idx1), pred=grep (idx0)
        {"state": "s", "question": "q", "options": ["grep", "read"], "target": {"read": 1.0}, "kind": "choice", "dp": "tool_select"},
    ]
    res = evaluate(_FakeEngine(), exs, n_perm=0)
    o = res["overall"]
    assert o["n"] == 4
    assert o["acc"] == pytest.approx(0.25)          # only the match is label-correct
    assert o["asym_valid"] == pytest.approx(0.5)     # match + forgiven reply_to_tool
    assert o["quad_match"] == pytest.approx(0.25)
    assert o["quad_reply_to_tool"] == pytest.approx(0.25)
    assert o["quad_tool_to_reply"] == pytest.approx(0.25)
    assert o["quad_other"] == pytest.approx(0.25)


def test_quadrant_idx_matches_name_quadrant():
    """The index-based twin must agree with the name-based quadrant on every combination."""
    opts, ri = ["read", "reply_without_tool", "grep"], 1
    for g in range(len(opts)):
        for p in range(len(opts)):
            assert S.quadrant_idx(p, g, ri) == S.quadrant(opts[g], opts[p])
    # explicit spot checks
    assert S.quadrant_idx(0, 0, 1) == "match"
    assert S.quadrant_idx(0, 1, 1) == "reply_to_tool"   # gold=reply, pick tool
    assert S.quadrant_idx(1, 0, 1) == "tool_to_reply"   # gold=tool, pick reply
    assert S.quadrant_idx(0, 1, -1) == "other"          # no reply option present


def test_reward_for_quadrant_and_reply_asym_reward():
    assert S.reward_for_quadrant("match") == 1.0
    assert S.reward_for_quadrant("reply_to_tool") == 0.0
    assert S.reward_for_quadrant("tool_to_reply") == -1.0
    assert S.reward_for_quadrant("other") == 0.0
    # index-based reward mirrors signed asymmetric_validity (readonly bonus off)
    assert S.reply_asym_reward(0, 0, 1) == 1.0    # match
    assert S.reply_asym_reward(0, 1, 1) == 0.0    # reply_to_tool forgiven
    assert S.reply_asym_reward(1, 0, 1) == -1.0   # tool_to_reply penalized
