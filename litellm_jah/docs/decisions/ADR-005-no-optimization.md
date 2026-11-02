# ADR-005: No performance optimizations at the measured load

Status: Accepted
Date: 2026-11-02

## Context

Code review flagged the plugin's theoretical hot paths: the journal appends synchronously under an
asyncio lock, `render_state`/`episode` re-scan the transcript (suspected O(n²)) on every decision,
and the session store's eviction scan is O(n). Before optimizing, a load test was run against a
real LiteLLM proxy fronting the production model (Qwen3.6-27B-FP8 via a local vLLM relay).

## Decision

Keep the simple, synchronous implementations. Measured on 2026-11-02:

| measurement | result |
|---|---|
| 60-request load (40 + 20 concurrent sessions) | 60/60 OK; e2e p50 ≈ 4–5 s (dominated by the 27 B model's generation) |
| log consistency after the load | every step exactly 1 routing row + 1 step row; zero lost rows |
| proxy memory | +28 MB RSS for 60 new sessions; no leak pattern |
| journal throughput (micro-benchmark) | 23–24 k rows/s single writer ≈ 4000× headroom over 5.6 req/s |
| `render_state` | < 1 ms up to 500 transcript steps (~0.9 ms at 500) |
| session store | < 1 ms for 10k gets; filling 5000 sessions 6 ms |

The plugin's own overhead is sub-millisecond at this load; the end-to-end latency is entirely the
upstream model.

## Alternatives Considered

- **Async/batched journal writes**: more moving parts (flush policy, ordering) for a path with four
  orders of magnitude of headroom. Deferred.
- **Incremental transcript rendering**: the O(n²) cost is only perceptible around ≥ 1000 steps; no
  known deployment hits that. Deferred.

## Consequences

- Revisit if either (a) proxy throughput grows ~100×, or (b) very long single-agent traces
  (> ~500 steps) become common — both are cheap to measure with the same harness used here.
- The load-test method (wave of concurrent sessions through the real proxy + journal consistency
  check + RSS check) is the standing regression reference for future performance work.
