# litellm-jah: JevAgentHarness as a LiteLLM proxy plugin

The routing decision point of [JevAgentHarness](../README.md) — a small, fast decision model that decides, per
agent step, whether the step is written by the strong LLM or the fast one — as a LiteLLM proxy plugin.

The engine is a port of `src/jevagentharness` (MIT): the same routing question, the same state rendering
(exact prompt format the decision models were trained on), the same policy (first steps strong, the p(hard)
cascade at `tau`, stick after an escalation, fast-streak cap) and the same fail-safe (a decision-model
failure never stops the agent: the step goes to the `on_error` tier, strong by default).

**Scope (P3):** all five decision points of the standalone proxy that a gateway can offer: routing
(`cascade` / `shadow` / `guard` / `off`), tool select, stop (including the finish check), reflection and the
action guard. `tool_params` and the memory points are not offered by a gateway at all. Budget, spend
tracking, auth and provider handling are LiteLLM's job, not this plugin's.

### Docker Compose

`docker/docker-compose.yml` runs the proxy with a shared Redis session store (multi-replica ready):

```bash
cd litellm_jah/docker
LITELLM_MASTER_KEY=sk-... DASHSCOPE_API_KEY=... OPENAI_API_KEY=... OPENAI_BASE_URL=... \
  docker compose up -d --build
```

Credentials are read from the environment or a `.env` file next to the compose file; an empty
`LITELLM_MASTER_KEY` generates a random key printed to the container logs. The journals are written
to `$HOME/.jah/litellm-logs` on the host (override with `JAH_LOG_DIR`); mount your own
`jah-config.yaml` / `litellm_config.yaml` as shown in the file's header.

## Requirements

- **LiteLLM v1.94+ has native routing plugins** — this package uses them: the routing-plugin wiring works on
  v1.94+, the complexity-router classifier-plugin wiring (recommended) on v1.99+.
- Python >= 3.10. Dependencies: `aiohttp`, `pyyaml` (no litellm import at all — the plugin duck-types the
  routing context, so the engine is unit-testable without a LiteLLM install).

```bash
pip install -e .            # from this directory
```

## Wiring 1 (recommended): classifier plugin on the complexity router

The classifier answers the routing question per request and returns the tier label; the complexity router
serves that tier's models. A classifier failure/timeout falls back to the router's `default_model` — set it
to the strong model to keep the JAH fail-safe. In litellm 1.106.x that also needs
`classifier_fallback: default_model` in the `complexity_router_config`: without it, a classifier
failure/timeout is handled by the built-in heuristic scorer instead, which can land a "simple"-looking
prompt on the fast tier.

`litellm_config.yaml`:

```yaml
model_list:
  - model_name: jah-auto
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        classifier_type: custom
        classifier_plugin: litellm_jah.classifier.jah_classifier   # or a file next to this config
        classifier_plugin_timeout_ms: 9000                          # > decision_timeout_s (8000) + margin
        tiers:
          COMPLEX: [jah-strong]
          SIMPLE: [jah-fast]
        default_model: jah-strong                                   # fail-safe: strong on classifier failure
  - model_name: jah-strong
    litellm_params:
      model: gemini/gemini-4-argon
      api_key: os.environ/GEMINI_API_KEY
  - model_name: jah-fast
    litellm_params:
      model: gemini/gemini-3.5-flash-lite
      api_key: os.environ/GEMINI_API_KEY
litellm_settings:
  callbacks:
    - litellm_jah.precall.jah_pre_call               # tool select / reflection / stop trigger, see Decision points
    - litellm_jah.postcall.jah_post_call             # finish check, stop guard, tool guard, routing guard, action guard
    - litellm_jah.logger.jah_step_logger             # JAH steps.jsonl (for `jah report`), see Logging
```

Start with `litellm --config litellm_config.yaml`. Clients send `model: "jah-auto"`; the `jah-strong` and
`jah-fast` groups remain directly addressable, like the standalone proxy's model names.

Notes:

- `classifier_plugin` is resolved as a dotted path at startup. Name the module singleton
`litellm_jah.classifier.jah_classifier` (as above) or the class `...classifier.JahClassifier` — either way it
runs on the single process-wide shared engine (an explicit `cfg`/`router` argument builds a dedicated one).
`pip install` makes both importable anywhere; otherwise drop a small `plugins/` file next to the config
(`from litellm_jah.classifier import jah_classifier`).
- **Do not enable `session_affinity`** for `jah-auto`: JAH re-decides the tier on every step, and
  session-affinity pins the first request's deployment for the rest of the session.
- The tier labels come from the plugin config (`strong_label` / `fast_label`); the `tiers:` keys above must
  match them.

## Wiring 2: routing plugin (narrows one model group)

For LiteLLM >= 1.94 without the complexity router: put both tiers in one model group and let the plugin
narrow the candidate pool to the chosen tier's models (`strong_models` / `fast_models` in the plugin config):

```yaml
model_list:
  - model_name: jah-auto
    litellm_params:
      model: gemini/gemini-4-argon
      api_key: os.environ/GEMINI_API_KEY
  - model_name: jah-auto
    litellm_params:
      model: gemini/gemini-3.5-flash-lite
      api_key: os.environ/GEMINI_API_KEY
router_settings:
  plugins: [litellm_jah.classifier.jah_routing_plugin]
  routing_strategy: simple-shuffle
litellm_settings:
  callbacks:
    - litellm_jah.precall.jah_pre_call      # tool select / reflection / stop trigger: shape the request
    - litellm_jah.postcall.jah_post_call    # finish check, stop guard, tool guard, routing guard, action guard
    - litellm_jah.logger.jah_step_logger    # JAH steps.jsonl (for `jah report`), see Logging
```

 ## Plugin config

 The module-level instances read their config at import time: `$LITELLM_JAH_CONFIG` (path to a YAML file),
 else `~/.jah/litellm.yaml` if present, else built-in defaults. See `config.example.yaml`. Import has two
 side effects to know: it builds the decision backend, and when `log_dir` is set it creates that directory and
 the two `.jsonl` files — an unwritable `log_dir` fails proxy startup at import, before the port is bound.
 Keys may sit flat at the top level or in a `router:` block; a flat key overrides the block's value only when
 it is not null (a flat `key: null` leaves the block value in place). The keys are the standalone proxy's
 `router:` block (so a JAH config's `router:` section can be reused as-is):

| key | default | meaning |
|---|---|---|
| `decision` | `{type: rules}` | the decision backend: `{type: jev, provider: typesafe\|openrouter\|local}` or `{type: rules}` (no key) |
| `tau` | `0.75` | escalate to the strong tier when p(hard) >= tau |
| `first_steps_strong` | `1` | the opening step(s) plan the work on the strong tier |
| `stick_strong` | `0` | after an escalation, keep this many further steps strong |
| `max_fast_streak` | `0` | >0: force a strong step after this many fast steps in a row |
| `mode` | `cascade` | `cascade` \| `shadow` (ask, log, everything strong) \| `guard` (the fast tier writes the step; after the call the decision model can re-do it on strong) \| `off` (everything strong, no question) |
| `on_error` | `strong` | tier when the decision model fails or times out |
| `verify_finish` | `true` | the finish check: a fast-tier final reply (no tool call) is re-run on the strong tier |
| `strong` | `null` | model name for the re-run calls; without it the redo checks are skipped (see Decision points) |
| `preset` | `null` | `route` \| `paper` \| `safe` \| `shadow` — a decision-point preset (see Decision points) |
| `decisions` | `null` | per-point overrides on top of the preset, e.g. `{tool_select: {mode: shadow}, stop: {mode: guard}}` |
| `strong_api_key` / `strong_api_base` | `null` | credentials for the re-run, when not in the environment (`os.environ/NAME` reads the variable at call time) |
| `decision_timeout_s` | `8` | timeout of one decision call (retried twice inside the Jev backend) |
| `strong_label` / `fast_label` | `COMPLEX` / `SIMPLE` | tier labels returned to the complexity router |
| `strong_models` / `fast_models` | `()` | provider/model strings per tier (wiring 2 only) |
| `max_sessions` / `session_ttl_s` | `5000` / `3600` | session store: the in-process cache cap and the idle TTL |
| `session_store` | `null` | `{type: redis, url: redis://host:port/db}` for a shared store across replicas / workers; optional `prefix` |
| `log_dir` | `null` | set (e.g. `~/.jah/litellm-logs`) to write the JAH log files, see Logging |
| `model_name` | `jah-auto` | the model name recorded per step (the group key of `jah report`) |


## Docker

The official LiteLLM image with the plugin installed. Build from the repository root:

```bash
docker build -f litellm_jah/Dockerfile -t litellm-jah:latest .
```

The image is self-contained: the proxy config (`/app/litellm_config.yaml`), a neutral JAH config
(`/app/jah-config.yaml`, referenced by the image's `LITELLM_JAH_CONFIG` env), and an entrypoint
that wraps the base image's and forwards extra arguments. The shipped model list is a template —
point `jah-strong` / `jah-fast` at your own models (mount a config, below).

### Quick start

Run it with your model credentials as environment variables (the shipped configs read only
`os.environ/` references — no keys in the image). The proxy master key comes from
`LITELLM_MASTER_KEY`; when it is unset the entrypoint generates a random one and prints it at
startup (recover it with `docker logs <container>`; set it to use a fixed key):

```bash
docker run -p 4000:4000 -e DASHSCOPE_API_KEY=... litellm-jah:latest --config /app/litellm_config.yaml
```

Verify with the master key:

```bash
curl -s localhost:4000/v1/models -H "Authorization: Bearer <master-key>"
```

### Deploying a container

A long-running deployment: fixed master key, your model credentials, your JAH config mounted in,
and the log directory on the host so `jah report` / `jah fit` work without copying files out:

```bash
docker run -d --name litellm-jah \
  -p 4000:4000 \
  -e LITELLM_MASTER_KEY=sk-your-fixed-master-key \
  -e DASHSCOPE_API_KEY=... -e OPENAI_API_KEY=... -e OPENAI_BASE_URL=... \
  -v $HOME/jah/jah-config.yaml:/app/jah-config.yaml:ro \
  -v $HOME/.jah/litellm-logs:/root/.jah/litellm-logs \
  litellm-jah:latest --config /app/litellm_config.yaml
```

- `-e LITELLM_MASTER_KEY` — the key clients authenticate with. Unset → the entrypoint generates a
  random one and prints it once at startup (not persisted).
- Model credentials are plain environment variables that the proxy config references via
  `os.environ/`; they never end up in the image.
- `-v …/jah-config.yaml:/app/jah-config.yaml:ro` — your JAH config (decision backend, modes,
  thresholds, `log_dir`). Because the image's default `LITELLM_JAH_CONFIG` is already
  `/app/jah-config.yaml`, a plain mount is enough. The shipped one is neutral: rules decision
  backend, no vendor endpoints; mount your own for a LLM decision backend or custom point modes.
- Same trick for the model list: `-v …/litellm_config.yaml:/app/litellm_config.yaml:ro` points
  `jah-strong` / `jah-fast` at your models; the entrypoint's `--config` argument then reads your
  file.
- The JAH logs land in the container's `~/.jah/litellm-logs` (the shipped config's `log_dir`),
  i.e. `/root/.jah/litellm-logs` in the image; mount that path and run `jah report` / `jah fit`
  on the host. An unwritable `log_dir` fails proxy startup at import, before the port is bound.
- Everything else is standard container operation (restart policy, resource limits); the base
  image's `prod_entrypoint.sh` runs the proxy.

### Multi-replica: shared Redis session store

More than one replica (or `num_workers > 1`) needs a shared session store: run Redis and point the
JAH config's `session_store` at it (the image ships the `redis` extra):

```yaml
# your mounted jah-config.yaml
session_store:
  type: redis
  url: redis://jah-redis:6379/0
```

```bash
docker network create jah
docker run -d --name jah-redis --network jah redis:7
docker run -d --name litellm-jah --network jah -p 4000:4000 \
  -e LITELLM_MASTER_KEY=sk-your-fixed-master-key \
  -e DASHSCOPE_API_KEY=... -e OPENAI_API_KEY=... -e OPENAI_BASE_URL=... \
  -v $HOME/jah/jah-config.yaml:/app/jah-config.yaml:ro \
  -v $HOME/.jah/litellm-logs:/root/.jah/litellm-logs \
  litellm-jah:latest --config /app/litellm_config.yaml
```

In-flight decisions stay process-local; a replica that cannot reach Redis serves the request with a
process-local session (no cross-replica continuity) rather than failing — see Sessions.

## Sessions

The plugin identifies the agent's run (session) from the request metadata: `jah_session`, `session_id` or
`litellm_session_id` (LiteLLM's `x-litellm-session-id` header lands in `metadata.session_id` when you use
session-aware features). Without an explicit id it hashes the system prompt and the first user message — the
standalone proxy's fallback, so the same conversation gets the same session in both. Session state (tier,
fast streak, pending escalations, the per-step decision cache for retries) lives in a `SessionStore`: an
in-process LRU by default, fine for one proxy replica. For a multi-replica proxy (or `litellm --num_workers N`)
set `session_store: {type: redis, url: …}` (needs the `redis` package, or the `redis` extra): each session is
one JSON blob in Redis, re-written when the step settles, with the idle TTL re-anchored on every write.
In-flight decisions and pending questions stay process-local (asyncio futures never cross processes); a
session read after its idle TTL is the state at its last settle, and two processes writing one session
concurrently are last-writer-wins (a conversation advances one step at a time, so this does not happen in
normal use). If the store cannot be reached (or the redis package is missing) the request still passes with a
process-local session — no cross-process continuity until it recovers: connection attempts are bounded
(1s connect / 2s read) and not retried for 30s after a failure (ADR-006).

## Decision points

The standalone proxy's five decision points, ported to the proxy's lifecycle: **routing** (the classifier /
routing plugin) and the four that shape or check a step's request and reply.

| point | question it asks | modes |
|---|---|---|
| `routing` | how hard is the agent's next step? | `cascade` (default) \| `shadow` \| `guard` (the fast tier writes the step; the decision model checks the reply and can re-do it on strong) \| `off` |
| `tool_select` | which tool should the agent call next? | `shadow` (log the answer) \| `assist` (shortlist the tool set) \| `cascade` (force the tool and/or the tier that writes its arguments) \| `guard` (check the tool the LLM chose) |
| `stop` | is ending the turn here right? | `strong` (default with `verify_finish`: the finish check) \| `guard` (ask, redo the reply on strong if not done) \| `shadow` (behave as `strong`, log for fitting) \| `trigger` (ask before the step; if the task is already done, note it and go strong) |
| `reflection` | should the agent pause and re-check? | `trigger` (ask before the step; if yes, a harness note and the strong tier) \| `shadow` \| `always` (a note every `every` steps, no question) |
| `action` | is the proposed action risky? | `shadow` (log) \| `guard` (check risky actions; a red flag re-does the step on strong with a caution note) |

Presets: `route` (the default: routing + finish check), `paper` (+ the paper's tool-choice cascade), `safe`
(+ the action guard), `shadow` (every point in shadow mode, to log labelled data for `jah fit`). A point's
config is `{mode: ..., tau: ..., ...}`; see `config.example.yaml` for the per-point keys (`k`, `min_tools`,
`max_options`, `tier`, `force`, `scope`, `submit`, `every`, `prefilter`, `tools`, `patterns`, `on_error`).

**Pre-call** (`jah_pre_call`): with the request, the questions that can still change the step are launched:
tool select, the stop trigger, a triggered reflection. The deciding ones (the stop trigger, a triggered
reflection, a tool-select assist/cascade) are awaited and applied to the request — harness notes on the last
message, a forced `tool_choice`, a shortlisted tool set — before the model runs; the non-deciding ones
(shadow, guard) run in parallel with the model call. A retry of the same step reuses its plan from the
session, never re-asking.

**Post-call** (`jah_post_call`): on the reply, the checks that apply run: the stop check (finish check /
guard / shadow), the tool-select guard, the routing guard, the action guard. Any check that wants a redo re-runs the step on
`strong` with its harness notes (one redo; one more look when a redo proposes a risky action, like the
standalone proxy), and its answer is labelled by what the redo did, so `jah fit` can fit its threshold.

Both hooks apply to **non-streaming chat completions routed by this plugin**, and both are fail-safe: a
decision-model failure or a failed strong re-run falls back to the LLM-only behaviour (the original reply is
kept — the deployment already succeeded, unlike the standalone proxy, where a failed strong re-run fails the
step). The hooks can never break a completed request (LiteLLM also isolates hook errors).

### The finish check (`verify_finish`, the default `stop` mode)

A fast-tier reply with no tool call would end the agent's turn, so the strong tier re-runs the same step from
the same request, and its reply is the one that ends the turn (or proposes the next tool call):

- It fires only for a step **this plugin routed to the fast tier**, on a **non-streaming chat completion**
  whose request **carried tools**, whose reply has **no tool call**, when `verify_finish` is on and `strong`
  is set.
- The re-run is a direct `litellm.acompletion` on `strong` with the same messages, tools and sampling
  parameters (the same pattern as LiteLLM's own skills hook), so the strong model's credentials must be
  available to the SDK: the environment, or `strong_api_key` / `strong_api_base` in the plugin config.
- On success the strong reply **replaces** the fast one, the session state is escalated exactly like the
  standalone proxy (`fast_streak` reset, `last_tier` strong, the step pinned strong for a retry), and the step
  row records **both calls** (fast + strong) with their separate latencies, tokens and costs.

`litellm_jah.hook.jah_finish_check` (the P2 name) is a backward-compatible alias of `jah_post_call`.

## Logging: `jah report` / `jah fit` on plugin runs

With `log_dir` set in the plugin config, the plugin writes the standalone proxy's log files in the exact schema
the `jah` CLI reads:

- **`decisions.jsonl`** — one row per decision call, in the `Ask.record()` shape (`p`, `choice`, `tau`,
`p_act`, `acted`, `effect`, `label`, `fit_ok`, latency, cost, error): the routing decision point is written
by the engine whenever the question is put to the decision model (cascade/shadow; forced steps are not
logged); one row per question of every other enabled point is settled by the post-call hook (and by the
logger on a failed request, so the questions that finished are logged, like the standalone proxy).
- **`steps.jsonl`** — one row per completed step (session, step, model name, the tier and reason the plugin
chose, tokens, latency, LLM cost via LiteLLM's cost map, the decision call's latency and cost). Written by the
`jah_step_logger` callback, so add it to `litellm_settings.callbacks` (both wiring examples above). Steps the
plugin did not route (other model groups) are not logged. A step with strong re-dos is one row with every
call in `calls` (first call, then each redo) and `reason: <the first redo's check>`, like the standalone
proxy.

```bash
jah report ~/.jah/litellm-logs   # per model name: sessions, steps, tier shares, cost, latency, errors
jah fit ~/.jah/litellm-logs      # threshold fitting + the per-point/mode describe table
```

Notes:

- The log directory is created with both files empty at startup, so `jah report` works from the first minute.
- `jah fit` fits the thresholds of the points it can label (`tool_select`, `stop`): run with their `shadow`
or `guard` modes first to collect labelled decisions, then fit. On a routing-only log it renders the
`routing:<mode>` describe table and notes that no labelled decisions exist yet. The routing `tau` stays the
configured one.
- Unpriced models (not in LiteLLM's cost map) are logged with `priced: false` and counted in
  `unpriced_calls`, like the standalone proxy.

## Tests

```bash
pip install -e ".[dev]"
pytest            # fully offline: scripted fake backends, no keys, no network
ruff check .
```

## Roadmap

- [x] **P1** decision logging in the JAH `decisions.jsonl` / `steps.jsonl` schema: `jah report` / `jah fit`
work on plugin runs.
- [x] **P2** the finish check (`verify_finish`) as a post-call hook: a fast-tier final answer re-done by the
strong tier.
- [x] **P3** the remaining decision points (tool select, stop guard, reflection, action guard) as pre/post-call
hooks.
- [x] **P4a** shared (Redis) session store for multi-replica proxies / multi-worker `litellm`.
- [x] **P4b** the fourth routing mode (`guard`: the fast tier writes the step, the decision model checks the reply).

## Decisions

The design decisions behind the plugin live in `docs/decisions/` (ADR-001 plugin form, ADR-002 the
shared-engine/deepcopy fix, ADR-003 log-schema alignment, ADR-004 fail-safe semantics,
ADR-005 the load-test verdict, ADR-006 the Redis session store).

## License

MIT (like the main project). The self-hosted `AgentDecision-4B` decision model is CC BY-NC 4.0; use the API
providers (`typesafe`, `openrouter`) or a model you have the rights to for commercial deployments.
