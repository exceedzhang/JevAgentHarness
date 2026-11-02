"""Re-entry guard: the check hooks call ``litellm.acompletion`` directly for their strong re-runs, and that call
goes through the same global callbacks as the original request. ``in_check`` marks such an internal call so the
hooks (and the step logger) ignore it instead of checking it again (and the logger instead of logging a second
row for the same step)."""

from __future__ import annotations

import contextvars

in_check: contextvars.ContextVar[bool] = contextvars.ContextVar("jah_in_check", default=False)

# The (session key, step) of the request currently being routed, set by ``JAHRouter.decide``. The
# success/failure logging events are dispatched in a new asyncio task that copies the request's
# context, so the step logger reads the routed identity from here: the event's own kwargs carry no
# request metadata, and the session key must match the one the classifier decided with.
current_step: contextvars.ContextVar[tuple[str, int] | None] = contextvars.ContextVar(
    "jah_current_step", default=None)
