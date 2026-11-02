"""The pure policy: forced tiers and the cascade split at tau."""

from __future__ import annotations

from litellm_jah.config import JAHCfg
from litellm_jah.policy import choose_tier, commit, forced_tier
from litellm_jah.sessions import Session


def test_forced_tier_rules():
    cfg = JAHCfg(first_steps_strong=1, stick_strong=0, max_fast_streak=0)
    s = Session(key="k")
    assert forced_tier(s, 0, cfg) == "first_steps"
    assert forced_tier(s, 1, cfg) is None
    s.stick_left = 2
    assert forced_tier(s, 5, cfg) == "stick"
    s.stick_left = 0
    cfg2 = JAHCfg(first_steps_strong=0, max_fast_streak=3)
    s.fast_streak = 2
    assert forced_tier(s, 5, cfg2) is None
    s.fast_streak = 3
    assert forced_tier(s, 5, cfg2) == "fast_streak"


def test_choose_tier_split_at_tau():
    cfg = JAHCfg(tau=0.75)
    assert choose_tier(0.74999, cfg) == ("fast", "routine", False)
    assert choose_tier(0.75, cfg) == ("strong", "hard", True)
    assert choose_tier(0.9, cfg) == ("strong", "hard", True)


def test_commit_streak():
    s = Session(key="k")
    commit(s, "fast")
    commit(s, "fast")
    assert s.fast_streak == 2 and s.last_tier == "fast"
    commit(s, "strong")
    assert s.fast_streak == 0 and s.last_tier == "strong"
