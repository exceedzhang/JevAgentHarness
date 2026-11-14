# A/B: AgentDecision-4B (JAH) vs Drex v1.5 (Nace.AI) on the JAH routing task

Decides, with evidence, whether the released **Drex v1.5** (9B foundation decision model) is a
drop-in replacement for **AgentDecision-4B** (4B LoRA) in JevAgentHarness's per-step routing loop.

## Why this is a fair test

Both models are *decision models* and both natively speak Jev's **System One** wire format
(`POST /v1/systemone`). So the A/B feeds the **identical request** to each model:

- the same agent `state` (the task + recent actions/results), from the bundled
  `research/ab_test/data/routing_gold_v2.jsonl` (292 held-out routing decisions, gold = `routine`/`hard`);
- the same routing **question** — JAH's `router.QUESTION` verbatim ("…is the next step `routine` or
  `hard`?…") with the same two option descriptions.

The only thing that differs between the two arms is the model that answers. The returned
`{routine: p, hard: p}` is scored by **one shared metric function** (`ab_common.score`), so the
numbers are directly comparable.

| Arm | Served by | Readout | New params |
|---|---|---|---|
| AgentDecision-4B | `research/.../serve_jev.Engine` (this repo) | LM-head gather over option-code tokens | 0 (LoRA on Qwen3-4B) |
| Drex v1.5 | `drex-decision-models/serve.py` | learned pointer head (`head.pt`) on Qwen3.5-hybrid 9B | ~2M |

## What it measures

- **routing accuracy** — `argmax(p) == gold` (the primary number);
- **calibration** — NLL, Brier, ECE;
- **the JAH tier policy** — strong-tier rate at `p(hard) ≥ 0.75`, and accuracy *among* strong-routed
  vs fast-routed decisions (the operational cost/quality split JAH actually uses);
- **per-step latency** — one decision per forward call, median / p95 / mean.

## Run it (on a CUDA machine)

From the repo root, on a box with an NVIDIA GPU and ~30 GB free:

```bash
bash research/ab_test/run_all.sh
```

That will: build the `research/.venv` (torch/transformers/peft), clone
`nace-ai/drex-decision-models` and build its venv, download both models, run both arms, and write
`research/ab_test/out/comparison.md` plus `jah.json` / `drex.json`.

Useful overrides:

```bash
LIMIT=50 bash research/ab_test/run_all.sh          # smoke test on 50 decisions first
SKIP_DREX=1 bash research/ab_test/run_all.sh       # JAH arm only
SKIP_JAH=1  bash research/ab_test/run_all.sh       # Drex arm only
```

Set `HF_TOKEN` if either repo is gated.

## Files

| File | Role |
|---|---|
| `ab_common.py` | routing question (verbatim from `router.py`), data loader, System One request builder, the shared `score()` |
| `run_jah.py` | JAH arm — loads AgentDecision-4B via `serve_jev.Engine`, one decision per call |
| `run_drex.py` | Drex arm — starts `serve.py`, one System One request per decision |
| `compare.py` | merges the two JSONs into `comparison.md` |
| `run_all.sh` | orchestrator (env + downloads + run + compare) |
| `data/` | bundled decisions — `routing_gold_v2.jsonl` (292, the A/B set) + `dev_tool_select.jsonl` (37); self-contained, no `research/data/` dependency |

The whole `research/ab_test/` directory is self-contained: copy it (plus the `research/src` package it
imports) to the GPU machine and run. Override the data with `DATA=/path/to/other.jsonl bash run_all.sh`.

## Reading the result

- If **Drex's `acc` ≥ JAH's** *and* its latency is acceptable, v1.5 is a viable (higher-generalization)
  replacement; check `strong_rate` — a model that over-routes to "hard" costs more in production.
- If **JAH's `acc` ≥ Drex's**, the 4B specialist wins its home task and the swap is not justified on
  routing alone (Drex may still be worth it for broader decision types, see `research/docs/DRex_vs_JAH_READOUT.md`).

## Caveats

- **Latency asymmetry**: the JAH arm is timed in-process (`Engine._probs`); the Drex arm is timed over
  localhost HTTP. Localhost overhead is ~1–2 ms, negligible next to tens-to-hundreds-of-ms decision
  latency, but it is a one-directional small bias. Both are the *reference serving path* of each model.
- **Prompt rendering differs** by construction (JAH: `A. routine -- desc`; Drex: `A: routine -- desc`
  + pointer head). The *semantic* content is identical; the rendering cannot be, because the
  architectures differ. That is precisely what the A/B isolates.
- **VRAM**: Drex v1.5 bf16 ≈ 18 GB. For ≤ 16 GB GPUs use the quantized `nace-ai/drex-v1.5-Q8_0`
  (pass `--weights .../drex-v1.5-Q8_0`); a small accuracy drop is possible.
- **Licensing**: AgentDecision-4B is CC BY-NC 4.0; Drex v1.5 is Nace.AI Open RAIL-M (stricter — see the
  commercial limits in the model card). This A/B does not change that.
