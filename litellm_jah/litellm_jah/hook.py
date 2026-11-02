"""Backward-compatible entry points for the finish check (its P2 name).

The finish check is the ``stop: strong`` / ``stop: trigger`` branch of the post-call checks: a fast-tier
final reply (no tool call) is re-run on the strong tier. ``JahFinishCheck`` / ``jah_finish_check`` are kept
so existing ``callbacks: [litellm_jah.hook.jah_finish_check]`` wiring keeps working; the full post-call
suite (finish check + stop guard/shadow + tool-select guard + action guard) is
``litellm_jah.postcall.jah_post_call``.
"""

from __future__ import annotations

from .postcall import JahPostCall, pass_through

__all__ = ["JahFinishCheck", "jah_finish_check", "pass_through"]


class JahFinishCheck(JahPostCall):
    """The P2 entry point — the full post-call suite, under its original name. With the default config
    (``stop: strong``, the other checks off) its behaviour is exactly the finish check; enabling
    tool_select / reflection / action in the JAH config activates those checks here too, exactly as in
    ``litellm_jah.postcall.jah_post_call``."""


jah_finish_check = JahFinishCheck()
