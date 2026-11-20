#!/usr/bin/env python
"""Dump ONE episode fully: task prompt, get_orders output, ask sequence, final action."""
import json, sys

path = "/data1/apps/JevAgentHarness/research/results/v8_decomp_n300/v8_failure_decomp_pa/v8_pa.jsonl"
ep = sys.argv[1] if len(sys.argv) > 1 else "clarify-156"

rows = [json.loads(l) for l in open(path) if l.strip()]
e = next((r for r in rows if ep in r.get("task_id","")), None)
if not e:
    print("not found"); sys.exit(1)

steps = e.get("steps", [])
# task prompt from the first decision point's full rendered question
q0 = None
for s in steps:
    for d in s.get("decisions", []):
        q0 = d.get("question") or d.get("state"); break
    if q0: break
print("=" * 10, "TASK / PROMPT (first 1400 chars)", "=" * 10)
print((q0 or "")[:1400])

print("=" * 10, "get_orders / find_user observations", "=" * 10)
for s in steps:
    if s.get("tool") in ("get_orders", "find_user"):
        obs = (s.get("observation") or "")
        print(f"[step {s.get('step')}] {s.get('tool')} {json.dumps(s.get('args'), ensure_ascii=False)[:120]}")
        print("  OBS:", obs[:1600])

print("=" * 10, "ask_user sequence (question -> customer reply)", "=" * 10)
for s in steps:
    if s.get("tool") == "ask_user":
        a = s.get("args") or {}
        q = a.get("question") or a.get("missing")
        print(f"[step {s.get('step')}] ask: {str(q)[:140]}  ->  {(s.get('observation') or '').strip()[:120]}")

print("=" * 10, "FINAL STATE-CHANGING ACTION", "=" * 10)
STATE = {"cancel_order","refund","ship_order"}
for s in steps:
    if s.get("tool") in STATE:
        print(f"[step {s.get('step')}] {s.get('tool')} args={json.dumps(s.get('args'), ensure_ascii=False)}")
        print("  OBS:", (s.get("observation") or "")[:300])

print("=" * 10, "RESULT", "=" * 10)
print(json.dumps({k:v for k,v in e["result"].items()}, ensure_ascii=False)[:300])
