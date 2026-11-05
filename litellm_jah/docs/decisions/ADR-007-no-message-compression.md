# ADR-007: Do not integrate message compression (Headroom)

Status: Withdrawn — reviewed, planned to v2, never implemented
Date: 2026-11-05

## Context

Headroom (a LiteLLM `CustomLogger` that compresses older/larger message content before the
upstream call) was evaluated as a candidate sixth decision point (`compress: off|shadow|on`)
to cut token cost further. A first plan was put through a fresh-context adversarial review
(27 findings, 11 HIGH); the review was largely accepted and produced a corrected plan (v2:
compression as a deterministic idempotent pure function of the full transcript). Before v2
was implemented, the work was deprioritized behind the decision-model switch and calibration,
and has now been withdrawn. No code, config, or ADR for compression ever reached the repo.

## Decision

Withdraw the compression integration. The load-bearing findings:

| issue | why it kills the plan as proposed |
|---|---|
| cache semantics inverted | the harness re-sends the **original** transcript every step; a proxy-side rewrite of the tail means step N+1's original history no longer matches step N's cached (compressed) tail — the prefix cache breaks at the previous tail **every step**. With the 10× cached discount, tail compression below ~90% is net-negative. |
| the economics are marginal anyway | ~95% of input tokens already bill as cached input at 1/10 price; a 30% compression of the remaining stream saves ~4–5% of input cost. The honest case was prefill latency and 393k context headroom, not ¥. |
| it pollutes the decision path | `render_state` sees the messages; compressed trajectories shift the routing/stop distributions and would confound the P-1 tau recalibration the 4B model needs (v2's mitigation — snapshot the state before transforming — added machinery for a change with no proven payoff). |
| accuracy risk is unmeasurable in live traffic | 10–20 canary sessions have no statistical power; proving "no accuracy regression" would require a fixed task-set benchmark with a pre-registered non-inferiority margin — a project of its own. |

## Consequences

- Message history is sent upstream as the harness wrote it. Token cost stays under control
  through the levers that are proven here: tier routing, the price table, and the cache hit
  rate already delivered by stable prefixes.
- The invariant this repo relies on — **upstream requests are byte-prefixes of the previous
  step's request** — stands unchallenged, and `jah fit` data keeps a single, stable meaning.
- Revisit only when a measured need appears: long-context prefill latency visible in user
  experience, or sessions hitting the context cap. If revisited, start from v2's core
  constraint (deterministic pure function of the full transcript, upstream = byte-extension
  of the last request), run it **after** tau recalibration is frozen, and measure on a fixed
  task set, not live traffic.
