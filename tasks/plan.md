# Implementation Plan: Decision-Model Eval → Asymmetric Validity / Outcome Reward

## Overview
Replace the label-match "56% gold-match" metric for the 4B decision model with a **validity-based, asymmetric** metric that encodes the §13.12 finding: extra/exploratory tool calls (reply→tool boundary) are **not penalized**, while premature replies / false-done (tool→reply boundary) **are heavily penalized**. Implemented as three layers: (1) an always-on cheap **rule-based** metric in the eval gate, (2) a mirrored **asymmetric GRPO reward**, (3) a reproducible **27B "reasonable-action" judge** as the periodic ground-truth.

## Why (the 56% artifact — `research/docs/DECISION_DATA_V2.md` §13.12)
- Current metric: `eval_decisions.py:74` `correct = (pred == gold)` → **56.4%** on v8 (`checkpoints/grpo_train_v8`, n=842 dev, strict argmax-vs-gold).
- Boundary analysis: **193** cases cross the reply↔tool boundary (exactly one of {gold,pred} is `reply_without_tool`).
  - **reply→tool** (51, model called a tool where gold=reply): **90.2% VALID** by the 27B judge → mostly *false* errors (extra read/verify is fine).
  - **tool→reply** (142, model replied where gold=tool): only **36.6% VALID / 62.7% INVALID** (抢答/假完成) → mostly *true* errors.
- The 27B judge scores **98/193** boundary cases VALID → "reasonable-action rate" = (475+98)/842 = **68.1%** (upper bound). So 56% **understates** real ability; the real weakness is execution-layer premature-reply, not the decision model.

## The asymmetric rule (core formula)
Let `RPL = "reply_without_tool"`; `is_reply(x) = (x == RPL)`.

| Case | Condition | Score (reward) | Acceptable? (eval) |
|---|---|---|---|
| Match | `gold == pred` | **+1.0** | yes |
| Extra tool (reply→tool) | `is_reply(gold) and not is_reply(pred)` | **0.0** | **yes (no penalty)** |
| Premature reply (tool→reply) | `not is_reply(gold) and is_reply(pred)` | **−1.0** | **no** |
| Other wrong tool | neither reply, `gold != pred` | **0.0** | neutral |

- **Eval headline** — `asym_valid` (a 0..1 fraction, comparable to `acc`): acceptable fraction = (matches + reply→tool) / total.
  - v8 → (475 + 51)/842 = **62.5%**. Sits between strict 56.4% (label) and the judge's 68.1% — a reproducible, LLM-free middle number.
- **GRPO reward** (scalar; needs sign + variance for the gradient): the signed Score column above (magnitudes tunable).
- Read-only bonus (give read-only extra tools +0.4 instead of 0.0) is **optional, off by default** (keeps it simple; matches "多调工具不罚" without a tool-specific list).

## Architecture Decisions
- **AD1 — Rule-based is primary & always-on** (no LLM cost); the 27B judge is periodic ground-truth, not the gate.
- **AD2 — Report, don't replace** `acc`: keep label-match `acc` for continuity; add `asym_valid` + the 4-quadrant counts alongside.
- **AD3 — Single source of truth**: one shared `asymmetric_validity(...)` helper used by both eval and reward (no drift).
- **AD4 — Backward compatible**: new reward name `reply_asym`; existing `--rewards correct,format` and all existing names unchanged.
- **AD5 — Reuse, don't rebuild**: `_READONLY_TOOLS` (move to shared), `boundary_extract.py`/`boundary_judge.py` (port into the package for the judge route), `mc_rollouts.py`/`grpo_online.py` (already the true outcome-reward route, noted not touched).

## Task List (vertical slices, ordered)

### Phase 0 — Foundation
- **Task 1** — Shared asymmetric scoring helper (`research/src/jevharness/training/scoring.py`): `asymmetric_validity(gold, pred, options) -> float`, `acceptable(gold, pred) -> bool`, `is_reply(tool)`, `quadrant(gold, pred) -> {match, reply_to_tool, tool_to_reply, other}`; move `_READONLY_TOOLS` here. Optional read-only bonus behind a flag.

### Checkpoint: Foundation
- [ ] Unit test covers all 4 quadrants + the read-only sub-case; expected values verified.

### Phase 1 — Eval gate (main deliverable, no GPU)
- **Task 2** — Wire the rule into `eval_decisions.py`: per-row `valid`/`acceptable`; add `asym_valid` + quadrant counts to `overall`/`by_dp`/`by_tool`; print alongside `acc`.

### Checkpoint: Eval gate
- [ ] `python -m jevharness.training.eval_decisions --model <base> --dev data/train_v4/dev.jsonl --limit 200` prints `asym_valid` + quadrants; a known tool→reply case → acceptable=no, reply→tool case → acceptable=yes.

### Phase 2 — GRPO reward mirror
- **Task 3** — Add `reply_asym` reward to `grpo_trl.py` (`make_reward_funcs` + `option_reward`): +1 gold, 0.0 reply→tool, −1.0 tool→reply, 0.0 other. Keep `reply_bound` unchanged.

### Checkpoint: Reward
- [ ] Unit test: gold=reply/pred=tool → 0.0; gold=tool/pred=reply → −1.0; gold==pred → +1.0.

### Phase 3 — Judge port (periodic ground-truth)
- **Task 4** — Port `boundary_extract.py` + `boundary_judge.py` into `training/boundary_validity.py` (a real module + CLI): `boundary_validity_rate(dev_predictions, judge_client)` → `{n, valid, by_dir, rate}`; reproduce §13.12 (≈193 boundary, ≈98 VALID, ≈68.1%).

### Phase 4 — Docs
- **Task 5** — Update `DECISION_DATA_V2.md` §13.12/§13.22: the `asym_valid` formula, the 4-quadrant table, the v8 numbers (56.4% / 62.5% / 68.1%), and how the rule-based proxy maps to the judge.

## Parallelization
- Task 1 (foundation) must precede 2 & 3. Tasks 2, 3, 4 are independent after 1 (can be done in parallel). Task 5 last.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| Tool-name coverage (customer-support tools ship/refund/cancel not in `_READONLY_TOOLS`) | Med | Make the **tool→reply penalty independent of read-only status** (it's a premature reply regardless of tool); keep reply→tool at 0.0 for *all* tools by default |
| 27B judge availability/cost (needs vllm `:8000` up; ~193+ calls/run) | Med | Judge is **optional/periodic** (Phase 3); the rule-based metric (Phases 0-1) is the always-on gate |
| Breaking the existing gate or training | Med | Report, don't replace `acc`; new reward name is additive (AD2/AD4) |
| Penalty magnitudes are a design choice | Low | Expose as CLI args with sensible defaults; confirm with user before locking (Open Q1) |
| `dev.jsonl` `target` is a `{option:prob}` dict; gold = argmax(target) | Low | Boundary detection uses `options` (names) — already available in eval & extract scripts |

## Open Questions (need your call)
1. **Penalty magnitudes** (defaults above): tool→reply = **−1.0** (抢答重罚), reply→tool = **0.0** (多调工具不罚), read-only bonus **off**. Proceed with these, or adjust (e.g. read-only +0.4)?
2. **Scope of this work**: just **Phase 0-1** (the eval metric — cheap, no GPU, literally what "修改评估" asked), or also **Phase 2** (GRPO reward) and **Phase 3** (judge port)? I recommend **0-1 now**, 2-4 as follow-on.
3. **Judge endpoint**: in-repo vllm `:8000` (the scripts' default) or the `:8001`/`:4001` 27B from `litellm_config.yaml`?
