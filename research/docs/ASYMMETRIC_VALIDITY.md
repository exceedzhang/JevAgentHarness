# Asymmetric Validity / Outcome Eval

Reference for the decision-model evaluation change requested after the §13.12 finding
("56% is largely a measurement artifact"). Full plan: `tasks/plan.md`.

## Why

The decision model (Qwen/Qwen3-4B, `dp: tool_select`) was scored with strict **label-match**
(`correct = (argmax == gold)`). §13.12 showed that on `n=842` dev (v8 = `checkpoints/grpo_train_v8`)
this understates real ability, and that the true weakness is **execution-layer** (premature reply /
false-done), not the decision model. So the eval now reports a **validity-based, asymmetric** number
alongside label-match, and the training reward can mirror the same rule.

## The rule (single source of truth: `training/scoring.py`)

Each decision is placed in one of four quadrants by comparing **gold** vs **pred**, where a
"reply" = the `reply_without_tool` option:

| Quadrant | gold → pred | `asymmetric_validity` | acceptable? |
|---|---|---|---|
| `match` | tool/tool (same) or reply/reply | **+1.0** | yes |
| `reply_to_tool` | gold=reply, pred=tool | **0.0** (forgiven) | yes |
| `tool_to_reply` | gold=tool, pred=reply | **−1.0** (heavy penalty) | no |
| `other` | tool/tool (different) | **0.0** (neutral) | no |

Asymmetry = the requested property: **extra tool calls are not penalized** (reply→tool forgiven),
**premature replies are heavily penalized** (tool→reply −1.0). `READONLY_TOOLS` / `READONLY_BONUS=0.4`
exist for an optional "reward read-only exploration" mode (off by default).

Key functions: `quadrant(gold,pred)`, `quadrant_idx(k,gold,reply_idx)` (index twin for rewards),
`asymmetric_validity`, `acceptable`, `reply_asym_reward(k,gold,reply_idx)`.

## Headline metric

```
asym_valid = (#match + #reply_to_tool) / n          # "acceptable" rate
```

| Metric | v8 (n=842) |
|---|---|
| `acc` (label-match, unchanged) | 56.4% (475/842) |
| **`asym_valid` (rule proxy)** | **62.5%** ((475+51)/842) |
| judge validity upper bound | 68.1% ((475+98)/842) |

`asym_valid` **sits between** strict label-match and the judge's bound. The rule forgives **all**
reply→tool; the 27B judge distinguishes valid (extra-verify) from invalid within that quadrant
(§13.12: 90.2% of reply→tool are valid). So the rule proxy is conservative (≤ judge).

## Where it lives

- **Eval gate** `training/eval_decisions.py` — every group (`overall`/`by_dp`/`by_kind`/`by_tool`)
  now reports `asym_valid` + the four `quad_*` shares next to `acc`. New flags: `--judge`,
  `--judge-api` (default `http://127.0.0.1:8000/v1/chat/completions`), `--judge-model`
  (default `Qwen/Qwen3.8-27B`), `--judge-key` (default `sk-1234`).
- **GRPO reward** `training/grpo_trl.py` — new `reply_asym` in `REWARDS` + `option_reward`,
  delegating to `scoring.reply_asym_reward`. Backward compatible (`correct`, `soft`, `qvalue`,
  `reply_bound`, `valid_alt` untouched).
- **Judge port** `training/judge.py` (from `scripts/boundary_judge.py`) — `judge_case`,
  `build_prompt`, `summarize`. `eval_decisions --judge` runs it on the `reply_to_tool` /
  `tool_to_reply` rows and reports a `judge` block (VALID rate overall + by direction).

## proxy → judge mapping

| Layer | Source | Role |
|---|---|---|
| `asym_valid` | rule (`scoring.py`) | always-on, no LLM, cheap; the **headline proxy** |
| `judge` | 27B `Qwen/Qwen3.8-27B` @ vllm `:8000` | periodic ground-truth; tells valid/invalid *within* each boundary quadrant |

## Run

```bash
# rule proxy (no GPU): overall/by_dp/by_tool get asym_valid + quad_*
python -m jevharness.training.eval_decisions <run> --limit 200

# + 27B judge on boundary rows (needs vllm :8000 up)
python -m jevharness.training.eval_decisions <run> --judge
```

Tests: `python -m pytest research/tests/test_scoring.py research/tests/test_judge.py` (16 tests).
