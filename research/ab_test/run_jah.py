"""JAH arm of the A/B: score AgentDecision-4B (Qwen3-4B + LoRA) on the routing decisions.

Uses the repo's own ``serve_jev.Engine`` -- the exact prompt, readout, calibration and knockout of the
served AgentDecision-4B -- so the numbers describe what the harness will see. Each decision is scored
as one single-item call (no cross-decision cache) so the measured latency is the per-step cost.

Run from the repo root (or set REPO_ROOT):
    python research/ab_test/run_jah.py \
        --data research/data/routing_gold_v2.jsonl \
        --base Qwen/Qwen3-4B \
        --adapter <downloaded>/AgentDecision-4B \
        --out research/ab_test/out/jah.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# make the research package importable (jevharness.*) regardless of the CWD
sys.path.insert(0, str(HERE.parent / "src"))

from ab_common import dump, jah_items, load_decisions, score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B arm: AgentDecision-4B on routing decisions")
    ap.add_argument("--data", required=True, help="routing decisions JSONL (gold + state + options)")
    ap.add_argument("--base", default="Qwen/Qwen3-4B", help="base model id or local dir")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (JevAgentHarness/AgentDecision-4B)")
    ap.add_argument("--calibration", default=None, help="calibration.json (default: <adapter>/calibration.json)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=5, help="untimed warmup decisions")
    ap.add_argument("--out", default=str(HERE / "out" / "jah.json"))
    a = ap.parse_args()

    from jevharness.training.serve_jev import Engine  # noqa: E402  (heavy import)

    cal = a.calibration or str(Path(a.adapter) / "calibration.json")
    if not Path(cal).exists():
        cal = None

    records = load_decisions(a.data)
    if a.limit:
        records = records[: a.limit]

    eng = Engine(a.base, a.adapter, cal, name="ab", max_len=a.max_len, batch_size=1)

    # warm up (CUDA kernels / tokenizer) without timing
    for r in records[: a.warmup]:
        eng._probs([jah_items(r)])

    results: dict[str, dict] = {}
    t_start = time.perf_counter()
    for i, r in enumerate(records):
        item = jah_items(r)
        t0 = time.perf_counter()
        probs_list = eng._probs([item])
        dt = (time.perf_counter() - t0) * 1000.0
        if i < a.warmup:
            continue
        p = dict(zip(r["options"], (float(x) for x in probs_list[0])))
        results[r["id"]] = {"probs": p, "latency_ms": dt}
    wall = (time.perf_counter() - t_start) * 1000.0

    metrics = score(records, results)
    metrics["wall_ms_all"] = round(wall, 1)
    metrics["model"] = a.base
    metrics["adapter"] = a.adapter
    metrics["calibration"] = cal
    dump({"metrics": metrics, "per": results}, a.out)
    print(__import__("json").dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
