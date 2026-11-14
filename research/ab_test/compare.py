"""Merge the two A/B arm outputs into one comparison table.

    python research/ab_test/compare.py \
        --jah research/ab_test/out/jah.json \
        --drex research/ab_test/out/drex.json \
        [--out research/ab_test/out/comparison.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (key, label, higher-is-better)  -- None = neutral
ROWS = [
    ("n", "n decisions", None),
    ("acc", "routing accuracy (argmax==gold)", True),
    ("nll", "NLL (lower better)", False),
    ("brier", "Brier", False),
    ("ece", "ECE (calibration)", False),
    ("mean_conf", "mean confidence", None),
    ("strong_rate", "strong-tier rate (p(hard)>=0.75)", None),
    ("acc_strong", "accuracy | routed strong", True),
    ("acc_fast", "accuracy | routed fast", True),
    ("latency_ms_median", "latency median (ms)", False),
    ("latency_ms_p95", "latency p95 (ms)", False),
    ("latency_ms_mean", "latency mean (ms)", False),
    ("wall_ms_all", "wall time all (ms)", False),
]


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}" if v < 10 else f"{v:.1f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jah", required=True)
    ap.add_argument("--drex", required=True)
    ap.add_argument("--out", default=str(HERE / "out" / "comparison.md"))
    a = ap.parse_args()

    jah = json.loads(Path(a.jah).read_text())["metrics"]
    drex = json.loads(Path(a.drex).read_text())["metrics"]

    lines = ["# A/B: AgentDecision-4B vs Drex v1.5 on JAH routing decisions", "",
             "Same System One request (agent state + JAH routing question, routine/hard) to both models; "
             "identical metric code. Higher accuracy is better; lower NLL/Brier/ECE/latency is better.", ""]
    lines.append("| Metric | AgentDecision-4B | Drex v1.5 | Δ (JAH−Drex) |")
    lines.append("|---|---|---|---|")
    for key, label, hib in ROWS:
        jv, dv = jah.get(key), drex.get(key)
        delta = ""
        if isinstance(jv, (int, float)) and isinstance(dv, (int, float)):
            d = jv - dv  # signed: JAH - Drex
            if d == 0:
                delta = "0"
            elif hib is None:
                delta = f"{d:+.4f}" if abs(d) < 10 else f"{d:+.1f}"
            else:
                jah_wins = d > 0 if hib else d < 0
                mark = "✅" if jah_wins else "⚠️"
                delta = (f"{d:+.4f}" if abs(d) < 10 else f"{d:+.1f}") + f" {mark}"
        lines.append(f"| {label} | {_fmt(jv)} | {_fmt(dv)} | {delta or '-'} |")

    # gold / pred distributions
    lines += ["", "## Distributions", ""]
    lines.append("| | AgentDecision-4B | Drex v1.5 |")
    lines.append("|---|---|---|")
    lines.append(f"| gold | {jah.get('gold_counter')} | {drex.get('gold_counter')} |")
    lines.append(f"| pred | {jah.get('pred_counter')} | {drex.get('pred_counter')} |")

    text = "\n".join(lines) + "\n"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
