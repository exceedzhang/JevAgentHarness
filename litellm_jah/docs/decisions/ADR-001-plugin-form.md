# ADR-001: JAH as a LiteLLM proxy plugin, not a separate proxy

Status: Accepted
Date: 2026-11-02

## Context

The standalone JevAgentHarness proxy already runs the decision points (routing, tool select, stop,
reflection, action) for agent traffic. Many deployments, however, already front their models with a
LiteLLM proxy (key management, fallbacks, session affinity, spend tracking). Running the JAH proxy
in front of or behind it would add a network hop, duplicate auth, and hide LiteLLM's routing and
affinity features from the decision points.

LiteLLM's extension points map one-to-one onto JAH's decision points: the complexity router's
classifier plugin runs before model selection (routing), pre-call hooks can reshape the outgoing
request (tool select / reflection / stop trigger), post-call hooks can re-run the reply (finish
check, stop/tool/action guards), and a `CustomLogger` observes the step (steps.jsonl).

## Decision

Implement JAH as the `litellm_jah` package: plugin instances that plug into an *existing* LiteLLM
proxy, plus a step logger. The engine (router, session store, journal, backends, decision points)
is written with **zero imports of litellm** — it duck-types every litellm object it touches, so the
whole decision engine is developed and tested offline (the 154-test suite runs with no network and
no litellm install).

## Alternatives Considered

- **Standalone JAH proxy chained in front of LiteLLM**: an extra hop on every request, duplicated
  keys, and LiteLLM's routing never sees JAH's tier decision (or vice versa). Rejected.
- **Fork/patch litellm**: the decision logic would live upstream of every proxy feature it wants to
  influence, and would rot against upstream releases. Rejected.
- **Sidecar calling JAH over an API**: cannot reshape the request before it reaches the model
  (pre-call decisions need to edit the outgoing payload), so the tool select / reflection / stop
  trigger points would be unimplementable. Rejected.

## Consequences

- The plugin inherits LiteLLM's lifecycle quirks — most notably the startup config deepcopy, which
  forced ADR-002.
- Every hook must be fail-safe: a plugin bug must never fail an agent request (ADR-004).
- The zero-import design keeps the engine unit-testable and reusable outside LiteLLM (the standalone
  proxy and the plugin share the same point/policy semantics).
