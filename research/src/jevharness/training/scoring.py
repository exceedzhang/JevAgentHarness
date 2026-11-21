"""Asymmetric decision validity (``DECISION_DATA_V2`` §13.12).

Scores a decision by ACTION TYPE, not by exact gold match. This is the single source of truth
shared by the eval gate (``eval_decisions``) and the GRPO reward (``grpo_trl``) so both use one
rule. Encodes the §13.12 finding: extra / exploratory tool calls (reply -> tool boundary) are NOT
penalized (多调工具不罚); premature replies / false-done (tool -> reply boundary) ARE heavily
penalized (抢答重罚).

Per case, ``gold`` / ``pred`` are option NAMES (``"name"`` or ``"name -- desc"``):

    quadrant        condition                        signed score   eval-acceptable
    ──────────────  ─────────────────────────────    ────────────   ─────────────────
    match           gold == pred                      +1.0           yes
    reply_to_tool   gold is reply, pred is a tool      0.0           yes (多调工具不罚)
    tool_to_reply   gold is a tool, pred is reply     -1.0           no  (抢答重罚)
    other           both tools, gold != pred           0.0           no  (a genuine wrong-tool miss)

``asymmetric_validity`` returns the signed score in [-1, 1] (for the reward). ``acceptable``
returns the eval-accept flag used by the ``asym_valid`` headline rate, which for v8 =
(matches + reply_to_tool) / n = (475 + 51) / 842 = 62.5% -- a reproducible, LLM-free number that
sits between the strict label-match (56.4%) and the 27B judge's "reasonable-action rate" (68.1%).
"""
from __future__ import annotations

REPLY = "reply_without_tool"

# Read-only / harmless tools (moved from ``grpo_trl._READONLY_TOOLS``). Used ONLY when
# ``readonly_bonus`` is enabled: a read-only extra tool then earns partial credit instead of 0.0.
READONLY_TOOLS = {
    "read", "read_file", "grep", "glob", "list", "search", "webfetch", "web_search",
    "shell", "shell_run", "context7_resolve-library-id", "context7_query-docs",
}

READONLY_BONUS = 0.4  # optional extra-credit for a read-only exploration (off by default)

_MATCH, _R2T, _T2R, _OTHER = "match", "reply_to_tool", "tool_to_reply", "other"


def option_name(opt: str) -> str:
    """Tool name of an option string (``"name"`` or ``"name -- desc"``)."""
    return str(opt).split(" -- ")[0].strip()


def is_reply(opt: str) -> bool:
    return option_name(opt) == REPLY


def quadrant(gold: str, pred: str) -> str:
    """Classify a (gold, pred) pair into one of ``match`` / ``reply_to_tool`` / ``tool_to_reply`` / ``other``."""
    if option_name(gold) == option_name(pred):
        return _MATCH
    if is_reply(gold) and not is_reply(pred):
        return _R2T
    if is_reply(pred) and not is_reply(gold):
        return _T2R
    return _OTHER


def asymmetric_validity(gold: str, pred: str, readonly_bonus: bool = False) -> float:
    """Signed asymmetric score in [-1, 1] (see module docstring)."""
    q = quadrant(gold, pred)
    if q == _MATCH:
        return 1.0
    if q == _R2T:
        return READONLY_BONUS if (readonly_bonus and option_name(pred) in READONLY_TOOLS) else 0.0
    if q == _T2R:
        return -1.0
    return 0.0


def acceptable(gold: str, pred: str) -> bool:
    """Eval-accept flag: acceptable iff the case is a gold match or a (forgiven) reply_to_tool."""
    return quadrant(gold, pred) in (_MATCH, _R2T)


# --------------------------------------------------------------- index-based twin (for rewards)
# The GRPO reward decodes a token id to an option INDEX, so it has (k, gold_idx, reply_idx) not
# names. These mirror ``quadrant`` / ``asymmetric_validity`` exactly for the equivalent inputs, so
# the training reward and the eval gate stay one rule (§13.12).
def quadrant_idx(k: int, gold: int, reply_idx: int) -> str:
    """Index-based quadrant. ``reply_idx`` = index of the reply option, or < 0 if the set has none."""
    if k == gold:
        return _MATCH
    if reply_idx >= 0 and k == reply_idx and gold != reply_idx:
        return _T2R      # gold is a tool, picked the reply   (抢答重罚)
    if reply_idx >= 0 and gold == reply_idx:
        return _R2T      # gold is the reply, picked a tool   (多调工具不罚)
    return _OTHER


def reward_for_quadrant(q: str) -> float:
    """Signed reward (§13.12): match +1.0, reply_to_tool 0.0, tool_to_reply -1.0, other 0.0."""
    return {_MATCH: 1.0, _R2T: 0.0, _T2R: -1.0, _OTHER: 0.0}[q]


def reply_asym_reward(k: int, gold: int, reply_idx: int) -> float:
    """Index-based twin of ``asymmetric_validity`` (readonly_bonus OFF) for the GRPO reward paths."""
    return reward_for_quadrant(quadrant_idx(k, gold, reply_idx))
