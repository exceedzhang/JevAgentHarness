# ADR-004: Fail-safe semantics — a decision failure never breaks the request

Status: Accepted
Date: 2026-11-02

## Context

The standalone proxy sits *behind* the agent's loop, so a failed strong re-run can legitimately fail
the step (the agent will retry). The plugin sits in the **hot path of every model call inside a
proxy**: a plugin bug, a decision-model timeout, or a bad response must never fail, or even
unnecessarily delay, the agent's request. This is a deliberate, documented divergence from the
standalone proxy.

## Decision

Fail-safe, at every layer:

- **Hooks never raise**: each pre/post-call hook and the logger wrap their whole body in
  `try/except` and return the unmodified request/response.
- **Decision calls are always bounded**: every backend call — including the rules backend and any
  custom `RulesBackend` subclass — runs under `asyncio.wait_for(decision_timeout_s)`; on failure the
  step takes the `on_error` tier (default: strong).
- **Checks are isolated**: post-call checks run concurrently with `gather(return_exceptions=True)`;
  one check that throws cannot starve the others or the settlement of the others' ask rows.
- **A failed re-run keeps the original reply** (`error_kept`), instead of failing the step like the
  standalone proxy.
- **Cancellation is cleaned up**: a cancelled request cancels its in-flight decision asks (nothing
  outlives the request), and the session store never evicts a session that has work in flight.
- **The routing plugin is fail-safe end to end**: any failure (including a routing context that
  lacks expected attributes) leaves the original candidate model list untouched.

## Alternatives Considered

- **Mirror the standalone's fail-closed redo**: a decision-model outage would then fail every
  fast-tier step in the proxy — unacceptable for a shared inference gateway. Rejected.
- **Per-check `try/except` without isolation**: one check's exception would still abort the rest of
  the post-call and drop the settled ask rows. Superseded by the isolated-gather design.

## Consequences

- The plugin logs **more** error rows than the standalone in the same failure (it keeps the ask
  rows the standalone would discard); fit analysis treats `error` rows as such.
- The divergences (redo-failure behaviour, error-row retention) are documented in the README so
  nobody has to rediscover them.
- The whole matrix is pinned by the offline test suite (isolation, cancellation, eviction,
  malformed-context tests), so the invariant survives refactors.
