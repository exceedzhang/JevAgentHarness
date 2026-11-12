#!/usr/bin/env python3
"""Fit a dedicated ``choice:2`` temperature for the (new-era) routing decision head, against the routing
gold produced by ``routing_gold_judge.py``.

The plugin applies its per-bucket temperature as ``p' = p^(1/T)/Z`` == ``softmax(log p / T)``; the
codebase's ``fit_temperature`` uses the identical transform (``softmax(log(p)/T)`` on raw probability
distributions), so the fitted T is exactly the value that belongs in the ``choice:2`` calibration key.

Method:
  1. invert the logged (calibrated) p to raw (T=1):  ``raw_i = logged_i^T_old / sum_j logged_j^T_old``
     (routing is 2-choice and currently rides the ``global`` temperature T_old);
  2. fit ``choice:2`` = ``fit_temperature(raw_dists, gold_idx)``;
  3. re-apply T to raw -> new calibrated p(hard); compare old vs new on NLL/Brier/ECE, firing rate,
     routing accuracy vs gold, AUC, and the "always routine" baseline.

A fit that hits the search bound (T->20) and collapses p(hard) to the prior is a NEGATIVE result: it
means the head has no usable (monotone) difficulty signal, so a temperature cannot fix it.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]   # research/
sys.path.insert(0, str(ROOT / "src"))
from jevharness.bench import metrics as M
from jevharness.decision.calibration import fit_temperature


def nll(dists, gold):
    tot = 0.0
    for p, g in zip(dists, gold):
        z = np.log(np.clip(p, 1e-12, 1.0))
        z = z - z.max()
        tot -= (z[g] - math.log(np.exp(z).sum()))
    return tot / max(1, len(gold))


def auc(score, label):
    pos = score[label == 1]
    neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    win = sum(p > n for p in pos for n in neg)
    tie = sum(p == n for p in pos for n in neg)
    return (win + 0.5 * tie) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "data/routing_gold_v2.jsonl"))
    ap.add_argument("--old-global", type=float, default=0.7720006824268275)
    ap.add_argument("--tau", type=float, default=0.85)
    args = ap.parse_args()

    rows = [json.loads(l) for l in pathlib.Path(args.gold).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("gold") in ("hard", "routine")]
    n = len(rows)
    hard = sum(1 for r in rows if r["gold"] == "hard")
    print(f"gold rows: {n}  (HARD={hard}={hard/n:.1%}, ROUTINE={n-hard}={(n-hard)/n:.1%})")

    ph = np.array([r["p_logged"]["hard"] for r in rows])
    pr = np.array([r["p_logged"]["routine"] for r in rows])
    g = np.array([1 if r["gold"] == "hard" else 0 for r in rows], dtype=float)

    # 1. invert logged (calibrated under T_old) -> raw
    OG = args.old_global
    raw_h = (ph ** OG) / ((pr ** OG) + (ph ** OG))
    raw_r = 1.0 - raw_h
    raw = np.stack([raw_r, raw_h], axis=1)
    gold_idx = [1 if r["gold"] == "hard" else 0 for r in rows]

    # 2. fit choice:2 temperature
    T = fit_temperature([raw[i] for i in range(n)], gold_idx)
    print(f"\nfit choice:2 temperature T = {T:.6f}   (search bound [0.05, 20]; T at a bound = negative result)")

    # 3. re-apply T -> new calibrated p(hard)
    e = 1.0 / T
    new_h = (raw_h ** e) / (raw_r ** e + raw_h ** e)
    new = np.stack([1.0 - new_h, new_h], axis=1)
    old = np.stack([pr, ph], axis=1)

    def quant(a):
        a = np.sort(a)
        return a[0], a[len(a) // 2], a[int(len(a) * 0.9)], a[-1]

    print("\n--- p(hard) distribution (min/p50/p90/max) ---")
    o, w = quant(ph), quant(new_h)
    print(f"old (global {OG:.3f}): min {o[0]:.3f} p50 {o[1]:.3f} p90 {o[2]:.3f} max {o[3]:.3f}")
    print(f"new (choice:2 {T:.3f}): min {w[0]:.3f} p50 {w[1]:.3f} p90 {w[2]:.3f} max {w[3]:.3f}")

    print("\n--- old vs new (lower metric = better; accuracy vs gold) ---")
    print(f"{'':10}{'NLL/s':>9}{'Brier':>9}{'ECE':>9}{'AUC':>7}  {'fires@tau':>10}  {'acc@tau':>8}  {'tau->37%':>9}")
    for name, dist, conf in (("old", old, ph), ("new", new, new_h)):
        a = auc(conf, g)
        srt = np.sort(conf)[::-1]
        tau37 = float(srt[int(0.37 * len(conf))])
        print(f"{name:10}{nll([dist[i] for i in range(n)], gold_idx):9.4f}"
              f"{M.brier(conf, g):9.4f}{M.ece(conf, g, n_bins=10):9.4f}{a:7.3f}  "
              f"{float(np.mean(conf >= args.tau)):10.1%}  {float(np.mean((conf >= args.tau).astype(int) == g)):8.3f}  {tau37:9.3f}")

    base = 1.0 - g.mean()
    best_acc = max(((t, float(np.mean((ph >= t).astype(int) == g))) for t in np.linspace(0, 1, 101)), key=lambda x: x[1])
    print(f"\nbaseline 'always routine' acc = {base:.3f} | best-over-tau acc = {best_acc[1]:.3f} @tau={best_acc[0]:.2f} "
          f"({'above' if best_acc[1] > base else 'NOT above'} baseline)")


if __name__ == "__main__":
    main()
