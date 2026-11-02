# ADR-006: The Redis session store for multi-process proxies

Status: Accepted
Date: 2026-11-02

## Context

`litellm` proxies run multiple processes (`--num_workers N`, or several replicas behind a load balancer).
Each process imports the plugin and builds its own `JAHRouter`, so with the default in-memory store the
same conversation held in two processes carries two divergent session states: the tier policy (fast
streak, `stick_left`, the per-step decision cache) breaks across workers, a retry landing on a different
worker re-asks or re-plans the step, and the "same session, same key" guarantee only holds per process.
The journal's two `.jsonl` files already tolerate multi-process append (short lines), so the session
state was the remaining single-process assumption.

## Decision

A `RedisSessionStore` behind the same `SessionStore` interface, selected by
`session_store: {type: redis, url: …}` (optional `prefix`), with one JSON blob per session and the idle
TTL re-anchored (`EXPIRE`) on every write:

- **What is shared**: the durable session fields (tier, fast streak, `stick_left`, the per-step
  decision cache, plans, re-run calls, step points), written when the step settles
  (`postcall.settle`) — i.e. the blob always holds the state at the conversation's last completed step.
- **What is not shared**: `deciding` and `pending` are in-process asyncio futures. Each process keeps a
  local LRU cache (same cap, same in-flight protection as the memory store) so concurrent requests in one
  process share one `Session` object; another process evicting or reloading the session cannot orphan an
  in-flight decision or question.
- **Fail-safe**: the Redis client is created lazily on first use with bounded timeouts (1s connect /
  2s read), so a misconfigured store never blocks proxy startup and can never stall a request. When the
  store is unreachable (or the `redis` package is missing), `get` falls back to a process-local session and
  opens a 30s dead window during which redis is not retried and `save` is skipped: the store is
  observability state, not the data path — a dead store degrades JAH to process-local routing, the request
  still passes. The `settle` save is also wrapped, so a write failure never breaks the request.

## Alternatives Considered

- **Per-field Redis hashes with distributed locks**: more round-trips and a lock protocol for a value
  (one session blob) that is rewritten atomically as a unit anyway.
- **LiteLLM's `session_affinity` / sticky routing**: would pin a conversation to one deployment,
  contradicting the complexity router's per-step tier decision (README already warns against it), and it
  does not cover the multi-replica case.
- **Saving on every mutation** (decide, precall, postcall): up to four writes per step for a state that
  is only observable at step boundaries; one save at settle is enough.
- **Staying single-process by mandate**: the simplest, and the design's default — but a multi-worker
  `litellm` is a common deployment, so the shared store is the supported escape hatch.

## Consequences

- `redis` is an optional dependency (the `redis` extra); the package's zero-litellm-import,
  offline-testable property is unchanged (tests run against a fake client).
- A process that crashes mid-step loses that step's state (the blob is the last settle); the
  conversation continues from there on any worker.
- Two processes writing one session concurrently is last-writer-wins. A conversation advances one step
  at a time, so this cannot happen in normal use; it is documented, not prevented.
- The in-memory store remains the default: single-replica proxies get exactly today's behavior.
