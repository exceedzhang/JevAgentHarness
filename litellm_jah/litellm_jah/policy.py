"""The JAH routing policy as pure functions (no I/O): the tier for a step, given the session state and the
decision model's p(hard). Mirrors ``jevagentharness.router.Router`` so the plugin's behaviour matches the
standalone proxy's.

Policy (``decide`` in :mod:`litellm_jah.engine`), for step t of a session:

1. the first ``first_steps_strong`` steps go to the strong tier (the agent reads the task and plans);
2. otherwise the decision model answers the routing question and the step goes to the strong tier iff
   p(hard) >= tau;
3. after an escalation the next ``stick_strong`` steps stay strong; with ``max_fast_streak`` > 0 a strong step is
   forced after that many fast steps in a row;
4. fail-safe: no answer (timeout, error) -> the ``on_error`` tier (strong by default).
"""

from __future__ import annotations

from .config import JAHCfg
from .sessions import Session

STRONG = "strong"
FAST = "fast"


def forced_tier(sess: Session, step: int, cfg: JAHCfg) -> str | None:
    """The reason a step goes to the strong tier without asking (``first_steps``, ``stick``, ``fast_streak``),
    else None. No side effects."""
    if step < cfg.first_steps_strong:
        return "first_steps"
    if sess.stick_left > 0:
        return "stick"
    if cfg.max_fast_streak > 0 and sess.fast_streak >= cfg.max_fast_streak:
        return "fast_streak"
    return None


def choose_tier(p_hard: float, cfg: JAHCfg) -> tuple[str, str, bool]:
    """(tier, reason, escalated): the cascade split at tau. ``escalated`` marks an escalation, after which
    ``stick_strong`` further steps stay strong."""
    if p_hard >= cfg.tau:
        return STRONG, "hard", True
    return FAST, "routine", False


def commit(sess: Session, tier: str) -> None:
    """Update the session's running state after a step's tier is settled."""
    sess.fast_streak = sess.fast_streak + 1 if tier == FAST else 0
    sess.last_tier = tier
