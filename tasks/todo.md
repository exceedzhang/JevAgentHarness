# Task List: Decision-Model Eval → Asymmetric Validity / Outcome Reward

Plan: `tasks/plan.md`. See Open Questions in the plan (penalty magnitudes, scope, judge endpoint) — **confirm before implementation**.

## Phase 0 — Foundation
- [x] Task 1: Shared asymmetric scoring helper `research/src/jevharness/training/scoring.py`
      `asymmetric_validity(gold,pred,options)->float`, `acceptable(gold,pred)->bool`, `is_reply(tool)`,
      `quadrant(gold,pred)`, move `_READONLY_TOOLS` here.
  - [x] gold==pred → +1.0 / acceptable
  - [x] reply→tool → 0.0 / acceptable (no penalty)
  - [x] tool→reply → −1.0 / not acceptable (heavy penalty)
  - [x] other wrong tool → 0.0 / neutral
  - [x] unit test `research/tests/test_scoring.py` covers all quadrants + read-only sub-case
- **Checkpoint**: `python -m pytest research/tests/test_scoring.py` passes

## Phase 1 — Eval gate (main deliverable, no GPU)
- [x] Task 2: Wire rule into `eval_decisions.py` — per-row `valid`; `asym_valid` + quadrant counts in `overall`/`by_dp`/`by_tool`, printed alongside `acc`
  - [x] `eval_decisions --limit 200` prints `asym_valid` + 4 quadrants
  - [x] known tool→reply case → acceptable=no; reply→tool case → acceptable=yes
  - [x] v8 reproduces ≈62.5% (475+51)/842 under this metric
- **Checkpoint**: eval gate runs clean on a zero-shot base model subset

## Phase 2 — GRPO reward mirror
- [x] Task 3: Add `reply_asym` reward to `grpo_trl.py` (`make_reward_funcs` + `option_reward`); keep `reply_bound` unchanged
  - [x] gold=reply/pred=tool → 0.0; gold=tool/pred=reply → −1.0; gold==pred → +1.0
  - [x] `--rewards correct,format` still works (backward compat)
- **Checkpoint**: reward unit test passes

## Phase 3 — Judge port (periodic ground-truth)
- [x] Task 4: Port `boundary_extract.py`+`boundary_judge.py` → `training/boundary_validity.py` (+CLI)
  - [x] `boundary_validity_rate(dev_preds, judge_client)` → {n, valid, by_dir, rate}
  - [x] reproduces §13.12 (≈193 boundary, ≈98 VALID, ≈68.1%) with the 27B judge; stub client for the unit test
- **Checkpoint**: judge runs over the dev set (needs 27B up) or passes the stub test

## Phase 4 — Docs
- [x] Task 5: Update `DECISION_DATA_V2.md` §13.12/§13.22 — `asym_valid` formula, 4-quadrant table, v8 numbers (56.4% / 62.5% / 68.1%), proxy→judge mapping

## Parallelization
- Task 1 precedes 2 & 3. Tasks 2/3/4 independent after 1. Task 5 last.
