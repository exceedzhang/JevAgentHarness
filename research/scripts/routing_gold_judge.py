#!/usr/bin/env python3
"""Build a routing gold set: label new-era routing states hard/routine with the strong model as a
reference judge, then join each label to its logged decision row (state + calibrated p) so the gold is
self-contained.

The strong model (the production strong tier) is the reference the router approximates, so "does the
strong model judge this state hard or routine" is the gold we calibrate the cheap decision model's
p(hard) against. The judge sees ONLY the state (not the outcome), matching the decision model's input.

Output row (one per routing decision)::

    {"id": "prod:<session>/<step>/routing", "dp": "routing", "kind": "choice",
     "options": ["routine", "hard"], "state": "...", "p_logged": {"routine": ..., "hard": ...},
     "calibrated": true, "truncated": false, "gold": "hard"|"routine",
     "judge": "deepseek-v4.1-flash", "judge_ts": <ts>}

Incremental/resumable: rows already in --out (by id) are skipped, so an interrupted run can be re-run.
The API key comes from the environment (never committed); --env-file loads a docker/.env-style file if
the key is not already exported.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]          # research/
REPO = ROOT.parent

PROMPT = (
    "Given ONLY the agent context below, classify the agent's NEXT (not yet made) step.\n"
    "ROUTINE = a fast, cheap, lower-capability LLM could handle it correctly (simple/straightforward "
    "tool call, short factual answer, low risk).\n"
    "HARD = it needs careful reasoning, complex multi-tool orchestration, resolving ambiguity/conflicts, "
    "or is high-risk/irreversible, so a fast model is likely to fail.\n"
    "Do not write long chain-of-thought. Your final line must be exactly: ANSWER: ROUTINE  or  ANSWER: HARD\n\n"
    "AGENT CONTEXT:\n%s"
)


def load_env_file(path: pathlib.Path) -> None:
    for line in path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_rows(decisions: pathlib.Path, since: str):
    rows = [json.loads(l) for l in decisions.read_text().splitlines() if l.strip()]
    cutoff = datetime.datetime.fromisoformat(since).timestamp()
    return [r for r in rows if r.get("point") == "routing" and float(r["ts"]) >= cutoff and r.get("state")]


def parse(blob: str):
    m = re.findall(r"ANSWER:\s*(ROUTINE|HARD)", blob, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m2 = re.findall(r"\b(ROUTINE|HARD)\b", blob, re.IGNORECASE)
    return m2[-1].upper() if m2 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default=str(pathlib.Path.home() / ".jah/litellm-logs/decisions.jsonl"))
    ap.add_argument("--since", default="2026-11-11 18:00", help="new-era cutoff, 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--out", default=str(ROOT / "data/routing_gold_v2.jsonl"))
    ap.add_argument("--env-file", default=str(REPO / "litellm_jah/docker/.env"))
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--base-url", default="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1")
    ap.add_argument("--key-env", default="DASHSCOPE_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    load_env_file(pathlib.Path(args.env_file))
    key = os.environ.get(args.key_env)
    if not key:
        sys.exit(f"set {args.key_env} (or pass --env-file) before running")
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=key)

    rows = load_rows(pathlib.Path(args.decisions), args.since)
    out = pathlib.Path(args.out)
    done_ids = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                done_ids.add(json.loads(l)["id"])
    todo = [r for r in rows if f"prod:{r['session']}/{r['step']}/routing" not in done_ids]
    print(f"total {len(rows)} | done {len(done_ids)} | todo {len(todo)}", flush=True)

    lock = threading.Lock()
    n = ok = fail = 0
    fout = open(out, "a")

    def work(r):
        nonlocal n, ok, fail
        label = None
        for attempt in range(1, 4):
            tail = "\n(Answer in one word only: ROUTINE or HARD.)" if attempt > 1 else ""
            try:
                d = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": PROMPT % r["state"] + tail}],
                    temperature=0, max_tokens=args.max_tokens)
                msg = d.choices[0].message
                blob = (msg.content or "")
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    blob = blob + "\n" + rc[-400:]
                label = parse(blob)
            except Exception:
                label = None
            if label in ("ROUTINE", "HARD"):
                break
        rec = {"id": f"prod:{r['session']}/{r['step']}/routing", "dp": "routing", "kind": "choice",
               "options": r["options"], "state": r["state"],
               "p_logged": {k: round(float(v), 6) for k, v in r["p"].items()},
               "calibrated": r.get("calibrated"), "truncated": r.get("truncated"),
               "gold": "hard" if label == "HARD" else "routine", "judge": args.model, "judge_ts": r["ts"]}
        with lock:
            n += 1
            ok += label in ("ROUTINE", "HARD")
            fail += label not in ("ROUTINE", "HARD")
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if n % 20 == 0:
                print(f"  {n}/{len(todo)} ok={ok} fail={fail}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
    fout.close()
    print(f"DONE: judged {n} new (ok={ok} fail={fail}); total rows in {out}:", flush=True)
    print(sum(1 for _ in out.open()), flush=True)


if __name__ == "__main__":
    main()
