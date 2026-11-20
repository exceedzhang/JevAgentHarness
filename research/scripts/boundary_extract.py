"""DCSV2 §13.12 -- extract the reply<->tool boundary cases from a decision-model checkpoint's dev predictions.

Runs the checkpoint over every held-out decision point (single stage, argmax) and records per-case
(gold_tool, pred_tool). A "boundary case" is one where exactly one of {gold, pred} is reply_without_tool,
i.e. the model crossed the reply<->tool boundary. Dumps the full per-case table and the boundary subset
(each with its full `state`) for the validity judge (boundary_judge.py).

Calibration is a temperature (softmax(logits/t)); it does NOT change the argmax, so the boundary set is
calibration-invariant.

  python boundary_extract.py --model <base> --adapter <ckpt> --dev <dev.jsonl> \
      --out-cases /tmp/boundary_cases.jsonl --out-percase /tmp/v8_dev_percase.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevharness.training.data import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data1/apps/huggingface/Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default="checkpoints/grpo_train_v8")
    ap.add_argument("--dev", default="data/train_v4/dev.jsonl")
    ap.add_argument("--out-cases", default="/tmp/boundary_cases.jsonl")
    ap.add_argument("--out-percase", default="/tmp/v8_dev_percase.jsonl")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    a = ap.parse_args()

    from jevharness.training.serve_jev import Engine

    RPL = "reply_without_tool"
    ex = [e for e in read_jsonl(a.dev) if 2 <= len(e.get("options") or []) <= 52 and e.get("target")]
    print("dev examples (2<=opts<=52, has target):", len(ex), flush=True)
    eng = Engine(a.model, a.adapter, None, name="ext", max_len=a.max_len, batch_size=a.batch_size)
    items = [(e["state"], e["question"], e["options"], e.get("kind", "choice")) for e in ex]
    probs = eng._probs(items)

    rows, conf = [], defaultdict(lambda: defaultdict(int))
    for e, p in zip(ex, probs):
        opts = e["options"]
        tgt = e["target"]
        gold_i = max(range(len(opts)), key=lambda j: float(tgt.get(opts[j], 0.0)))
        pred_i = max(range(len(p)), key=lambda j: p[j])
        gold, pred = opts[gold_i], opts[pred_i]
        conf[gold][pred] += 1
        rows.append({"id": e["id"], "opts": opts, "gold": gold, "pred": pred,
                     "state": e["state"], "question": e["question"], "kind": e.get("kind", "choice")})

    r2t = sum(n for g, pc in conf.items() if g == RPL for _, n in pc.items() if _ != RPL)
    t2r = sum(n for g, pc in conf.items() if g != RPL for _, n in pc.items() if _ == RPL)
    print(f"gold=reply->tool: {r2t}   gold=tool->reply: {t2r}   total boundary: {r2t + t2r}")

    boundary = []
    for r in rows:
        if (r["gold"] == RPL) != (r["pred"] == RPL):
            boundary.append({**r, "direction": "reply_to_tool" if r["gold"] == RPL else "tool_to_reply"})

    with open(a.out_percase, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "gold": r["gold"], "pred": r["pred"]}, ensure_ascii=False) + "\n")
    with open(a.out_cases, "w") as f:
        for b in boundary:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    bydir = defaultdict(lambda: defaultdict(int))
    for b in boundary:
        bydir[b["direction"]][b["pred"] if b["direction"] == "reply_to_tool" else b["gold"]] += 1
    for d in ("reply_to_tool", "tool_to_reply"):
        print(f"\n-- {d} ({sum(bydir[d].values())}) --")
        for t, n in sorted(bydir[d].items(), key=lambda x: -x[1]):
            print(f"   {t:30s} {n}")
    print("\nwrote", a.out_cases, len(boundary), "and", a.out_percase, len(rows))


if __name__ == "__main__":
    main()
