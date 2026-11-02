# ADR-003: Log schema aligned with the standalone proxy

Status: Accepted
Date: 2026-11-02

## Context

The main project's CLI (`jah report`, `jah fit`) analyzes decision quality from two JSONL files
written by the standalone proxy: `decisions.jsonl` (one row per decision call — routing, tool
select, stop, reflection, action) and `steps.jsonl` (one row per agent step — tier, calls, cost,
latency). For plugin runs to be analyzable by the same tooling — and to be comparable with
standalone runs in the same fit dataset — the plugin must emit the same shapes.

## Decision

The plugin writes byte-compatible rows:

- `decisions.jsonl`: `head` (`ts`, `session`, `model_name`, `step`) + the standalone `Ask.record()`
  fields per point (`point`, `mode`, `question`, `options`, `p`, `choice`, `confidence`, `tau`,
  `act_option`, `p_act`, `acted`, `effect`, `label`, `label_tier`, `agree`, `fit_ok`, `latency_s`,
  `cost_usd`, `model`, `rounds`, `error`, `extra`).
- `steps.jsonl`: the head + `tier`, `reason`, `p_hard`, `decision`, `calls` (the step's model
  calls, with the strong redo merged in as `redo_calls`), cost and token totals.

Where litellm's representations differ from the standalone's, the plugin normalizes: the recorded
model is the deployment's real model name (from the call kwargs, falling back to the response), and
latencies arriving as `datetime` objects are converted to seconds.

## Alternatives Considered

- **A plugin-specific schema plus a converter to the standalone shape**: two schemas to maintain,
  and fit data from the two sources would still need normalization at analysis time. Rejected.
- **Importing the standalone server's logging code into the plugin**: the plugin deliberately has
  zero hard dependencies on the standalone package (ADR-001); sharing code across that boundary
  would couple release cycles. Rejected — instead the shapes are kept compatible by construction
  and by review.

## Consequences

- Any schema change in the standalone proxy must be mirrored here (and vice versa); the review
  process treats the two as one schema with two writers. This has already paid off: the
  tool-select `fit_ok` / `after` semantics were aligned during the plugin's code review.
- Log writes are best-effort: a journal failure is swallowed (logged), never propagated to the
  request (ADR-004).
