"""DCSV2 §13.12 -- 27B validity judge over the reply<->tool boundary cases.

For each boundary case (from boundary_extract.py) it asks a 27B judge a PURE VALIDITY question
(not "is it gold", not "which is better"): is the action the agent actually chose a valid, reasonable,
non-contradictory action in this situation? Output per case: VALID / INVALID / UNCERTAIN + one-line reason,
plus an aggregate table by direction and by tool cell.

  python boundary_judge.py --in /tmp/boundary_cases.jsonl --out /tmp/boundary_judge.jsonl \
      --summary /tmp/boundary_judge_summary.json --api http://127.0.0.1:8000/v1/chat/completions \
      --key sk-1234 --model Qwen/Qwen3.8-27B
"""
import argparse
import json
import re
import urllib.request
from collections import defaultdict

RPL = "reply_without_tool"


def clip_state(state: str, head: int = 700, tail: int = 7000) -> str:
    s = state.strip()
    if len(s) <= head + tail + 20:
        return s
    return s[:head] + "\n...[middle elided]...\n" + s[-tail:]


def describe_choice(pred: str) -> str:
    if pred == RPL:
        return "a FINAL TEXT REPLY to the user (it chose to respond without calling any tool)"
    return f"calling the tool/action: {pred}"


def build_prompt(b: dict) -> str:
    lines = [
        "You are an expert auditor of an AI agent that picks exactly one action per decision point from a fixed list.",
        "Below is the exact situation: the task, the steps already taken with their results, the available actions, and the action the agent actually chose.",
        "YOUR ONLY JOB: judge whether the CHOSEN action is a VALID, REASONABLE action in this situation.",
        "  - VALID = a competent agent could plausibly and defensibly take it here; it is consistent with the task and the evidence gathered so far, and is not self-contradictory. It does NOT have to be the single best action; several actions may be valid at once.",
        "  - INVALID = clearly a mistake given the context: targets a wrong or nonexistent file, uses an irrelevant tool, repeats an action that already failed, or (for a final reply) the agent claims the task is done / gives an answer when the required information or work is plainly missing.",
        "  - UNCERTAIN = the context does not give enough signal to call it.",
        "Do NOT judge which action is better, and do NOT check whether it matches any reference answer. Judge only the validity of the chosen action itself.",
        "",
        "SITUATION:",
        clip_state(b["state"]),
        "",
        "AVAILABLE ACTIONS:",
    ]
    for i, o in enumerate(b["opts"]):
        lines.append(f"  {chr(65 + (i % 26))}{'' if i < 26 else i // 26}. {o}")
    lines.append("")
    lines.append(f"The agent CHOSE: {describe_choice(b['pred'])}")
    lines.append("")
    lines.append("Answer: first line exactly one of VALID / INVALID / UNCERTAIN, then a single short reason.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/tmp/boundary_cases.jsonl")
    ap.add_argument("--out", default="/tmp/boundary_judge.jsonl")
    ap.add_argument("--summary", default="/tmp/boundary_judge_summary.json")
    ap.add_argument("--api", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--key", default="sk-1234")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    a = ap.parse_args()

    def call(prompt: str) -> str:
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(a.api, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + a.key})
        with urllib.request.urlopen(req, timeout=180) as resp:
            m = json.loads(resp.read())["choices"][0]["message"]
        return (m.get("content") or "") or (m.get("reasoning_content") or "")

    def verdict(txt: str) -> str:
        m = re.search(r"\b(VALID|INVALID|UNCERTAIN)\b", txt, re.I)
        return m.group(1).upper() if m else "PARSE_ERR"

    cases = [json.loads(l) for l in open(a.inp)]
    print("boundary cases:", len(cases), flush=True)
    res = []
    with open(a.out, "w") as out:
        for i, b in enumerate(cases):
            try:
                ans = call(build_prompt(b))
                v = verdict(ans)
            except Exception as e:
                ans, v = f"ERR {e}", "ERR"
            res.append({"id": b["id"], "direction": b["direction"], "gold": b["gold"], "pred": b["pred"],
                        "verdict": v, "reason": ans.strip()[:300]})
            out.write(json.dumps(res[-1], ensure_ascii=False) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                done = defaultdict(lambda: defaultdict(int))
                for r in res:
                    done[r["direction"]][r["verdict"]] += 1
                print(f"[{i+1}/{len(cases)}] " + "  ".join(f"{d}:{dict(v)}" for d, v in done.items()), flush=True)

    def agg(direction=None):
        sel = [r for r in res if (direction is None or r["direction"] == direction)]
        n = len(sel)
        c = defaultdict(int)
        for r in sel:
            c[r["verdict"]] += 1
        return {"n": n, **{k: round(v / n, 3) for k, v in c.items()}, "raw": dict(c)} if n else {}

    cell = {}
    for r in res:
        key = r["gold"] if r["direction"] == "tool_to_reply" else r["pred"]
        cell.setdefault(r["direction"], {}).setdefault(key, {}).setdefault(r["verdict"], 0)
        cell[r["direction"]][key][r["verdict"]] += 1

    summary = {"all": agg(), "tool_to_reply": agg("tool_to_reply"), "reply_to_tool": agg("reply_to_tool"), "cells": cell}
    with open(a.summary, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("\n=== AGGREGATE VALIDITY (fraction VALID per direction) ===")
    for k in ("all", "tool_to_reply", "reply_to_tool"):
        print(f"{k}: {summary[k]}")
    print("written:", a.out, a.summary)


if __name__ == "__main__":
    main()
