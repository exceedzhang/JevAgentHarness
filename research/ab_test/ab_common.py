"""Shared core for the AgentDecision-4B vs Drex v1.5 routing A/B test.

Both models are decision models that speak Jev's System-One wire format (``POST /v1/systemone``).
The A/B feeds the IDENTICAL System One request -- the same agent state plus JAH's routing question
(routine vs hard) with the same option descriptions -- to each model and scores the returned
probabilities with the SAME metric code. The only thing that differs between the two arms is the
model that answers.

This module holds the routing question (copied verbatim from ``src/jevagentharness/router.py``),
the data loader, the System One request builder, and the scoring function. It has no dependency on
either model's runtime so it runs anywhere Python does.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ the routing question (JAH router.py, verbatim)
QUESTION_ID = "next_step"
QUESTION_TEXT = (
    "You are watching an AI agent work on the task below, one tool call at a time. Read its latest action and "
    "the result. Decide how hard the agent's NEXT step is. 'routine': the next step follows directly from what "
    "the agent already knows, e.g. read or search more, run a command or test it planned, make an edit it "
    "already worked out. 'hard': the next step needs careful reasoning, e.g. something failed or behaved "
    "unexpectedly, a plan or a fix must be worked out, the agent seems stuck or repeats itself, or it must "
    "judge whether the task is really done."
)
OPTIONS = ["routine", "hard"]
DESCRIPTIONS = {
    "routine": "The next step is routine: a faster, cheaper model will do it as well as a strong one",
    "hard": ("The next step needs careful reasoning: debugging, planning, recovering from an error, or "
             "verifying the result"),
}
HARD = "hard"
ROUTINE = "routine"
TAU = 0.75  # JAH routes to the strong tier iff p(hard) >= TAU

# JAH renders a wire Choice criterion as ``label -- description`` (training/data.py option_text).
JAH_OPTION = {o: f"{o} -- {DESCRIPTIONS[o]}" for o in OPTIONS}


def load_decisions(path: str | Path) -> list[dict[str, Any]]:
    """Read a decisions JSONL into records: {id, state, options, gold}.

    Accepts the repo's routing format (``gold`` + ``p_logged``) and the training/dev format (``target``).
    """
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        opts = r.get("options") or OPTIONS
        gold = r.get("gold")
        if gold is None:
            tgt = r.get("target") or {}
            if tgt:
                gold = max(tgt, key=lambda k: tgt[k])
        if gold is None:
            continue
        out.append({"id": r.get("id"), "state": r["state"], "options": opts, "gold": gold,
                    "p_logged": r.get("p_logged")})
    return out


def systemone_request(rec: dict[str, Any]) -> dict[str, Any]:
    """The System One request shared by BOTH arms: state + the routing choice question."""
    return {
        "state": rec["state"],
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": QUESTION_TEXT,
                # key order is the option order; values are the descriptions
                "criteria": {o: DESCRIPTIONS[o] for o in rec["options"] if o in DESCRIPTIONS},
            }
        },
    }


def jah_items(rec: dict[str, Any]) -> tuple[str, str, list[str], str]:
    """(state, question, options, kind) for ``serve_jev.Engine._probs``.

    Options are the wire form ``label -- description`` so the JAH model sees exactly what it was
    trained/served on. The returned probability list is in this same order.
    """
    opts = [JAH_OPTION[o] if o in JAH_OPTION else o for o in rec["options"]]
    return rec["state"], QUESTION_TEXT, opts, "choice"


# ------------------------------------------------------------------ metrics (identical for both arms)
def _ece(conf: list[float], corr: list[int], bins: int = 10) -> float:
    if not conf:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    tot = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == bins - 1 and hi <= c <= 1.0 and c <= hi + 1e-9)]
        if not idx:
            continue
        w = len(idx) / len(conf)
        acc = sum(corr[i] for i in idx) / len(idx)
        cf = sum(conf[i] for i in idx) / len(idx)
        tot += w * abs(acc - cf)
    return tot


def score(records: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score probabilities against gold. ``results[id] = {"probs": {option: p}, "latency_ms": float}``."""
    conf, corr, nll, per = [], [], [], []
    strong, fast = [], []
    lat = []
    n = 0
    for r in records:
        res = results.get(r["id"])
        if not res:
            continue
        probs = res["probs"]
        gold = r["gold"]
        p_gold = float(probs.get(gold, 0.0))
        pred = max(probs, key=lambda k: probs[k])
        p_pred = float(probs[pred])
        correct = int(pred == gold)
        conf.append(p_pred)
        corr.append(correct)
        nll.append(-math.log(max(p_gold, 1e-12)))
        if res.get("latency_ms") is not None:
            lat.append(float(res["latency_ms"]))
        # JAH tier policy: strong iff p(hard) >= TAU
        p_hard = float(probs.get(HARD, 0.0))
        (strong if p_hard >= TAU else fast).append(correct)
        per.append({"id": r["id"], "gold": gold, "pred": pred, "correct": correct,
                    "p_hard": p_hard, "tier": "strong" if p_hard >= TAU else "fast",
                    "conf": round(p_pred, 4)})
        n += 1
    if n == 0:
        return {"n": 0}
    acc = sum(corr) / n
    sr, fr = len(strong) / n, len(fast) / n
    def _acc(v): return (sum(v) / len(v)) if v else None
    out = {
        "n": n,
        "acc": round(acc, 5),                       # routing accuracy: argmax == gold
        "nll": round(sum(nll) / n, 5),
        "brier": round(sum((c - k) ** 2 for c, k in zip(conf, corr)) / n, 5),
        "ece": round(_ece(conf, corr), 5),
        "mean_conf": round(sum(conf) / n, 5),
        "strong_rate": round(sr, 5),                # share routed to the strong tier
        "acc_strong": round(_acc(strong), 5) if strong else None,  # accuracy on strong-routed
        "acc_fast": round(_acc(fast), 5) if fast else None,        # accuracy on fast-routed
        "n_strong": len(strong),
        "n_fast": len(fast),
        "latency_ms_median": round(sorted(lat)[len(lat) // 2], 2) if lat else None,
        "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2) if lat else None,
        "latency_ms_mean": round(sum(lat) / len(lat), 2) if lat else None,
        "gold_counter": dict(Counter(r["gold"] for r in records)),
        "pred_counter": dict(Counter(p["pred"] for p in per)),
    }
    return out


def dump(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")


if __name__ == "__main__":
    import sys
    recs = load_decisions(sys.argv[1])
    print(f"{len(recs)} decisions; gold:", dict(Counter(r["gold"] for r in recs)))
    print("System One request sample:\n", json.dumps(systemone_request(recs[0]), indent=1)[:600])
