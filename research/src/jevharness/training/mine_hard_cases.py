"""Mine hard decision cases from production logs or an existing labelled JSONL.

The queue is meant for human review / targeted data collection, not for automatic training: it
contains model errors, low-margin correct decisions, and shell-bias cases.  Each queue row keeps the
original prompt and label and adds the model's prediction, confidence and reasons.

Examples::

    # production log (the same journal used by data_v2)
    PYTHONPATH=src python -m jevharness.training.mine_hard_cases \
        --decisions ~/.jah/litellm-logs/decisions.jsonl \
        --model /data1/apps/huggingface/Qwen/Qwen3-4B \
        --adapter checkpoints/grpo_train_v3_clean_p1/step-000040 \
        --exclude data/train_v3_clean/train.jsonl data/train_v3_clean/dev.jsonl \
        --out /tmp/opencode/hard_cases

    # existing labelled data (useful for a smoke test / auditing a split)
    PYTHONPATH=src python -m jevharness.training.mine_hard_cases \
        --examples data/train_v3_clean/dev.jsonl --limit 20 \
        --model /data1/apps/huggingface/Qwen/Qwen3-4B \
        --adapter checkpoints/grpo_train_v3_clean_p1/step-000040 \
        --out /tmp/opencode/hard_cases_dev

The output is ``queue.jsonl`` plus ``summary.json`` and ``SUMMARY.md``.  ``queue.jsonl`` is sorted
P0 -> P2 and errors before low-margin cases; each ``(priority, gold tool)`` bucket is capped by
``--max-per-tool`` so a single dominant tool cannot fill the queue.  ``--tool-caps`` (JSON
``{tool: cap}``, string or file) overrides the per-bucket cap with a per-tool cap counted across
priorities (``0`` excludes the tool, e.g. to mine non-shell cases only), and ``--max-per-session``
caps rows per ``(session, tool)`` pair so one long trajectory cannot dominate a single tool,
while still letting it contribute to several different tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .data_v2 import from_production

# Targeted collection priorities from the P1 per-tool analysis.  Shell is deliberately P3: the
# model already predicts it well, and more easy shell rows would reinforce the default bias.
P0_TOOLS = {"reply_without_tool", "edit", "read", "grep", "glob", "write"}
P1_TOOLS = {"compress", "done", "not_done", "shell_run", "shell_wait"}
P2_TOOLS = {
    "webfetch", "websearch", "web_search", "web_fetch", "browse_web", "question",
    "ask_user_question", "ask_human", "present", "context7_resolve-library-id",
    "context7_query-docs", "firecrawl-mcp_firecrawl_search",
}
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def tool_of(ex: dict[str, Any]) -> str:
    """Gold tool = argmax of the target distribution."""
    target = ex.get("target") or {}
    if not target:
        return "?"
    return max(target, key=lambda k: float(target[k]))


def priority_of(tool: str) -> str:
    if tool in P0_TOOLS:
        return "P0"
    if tool in P1_TOOLS:
        return "P1"
    if tool in P2_TOOLS:
        return "P2"
    return "P3"


def decision_key(ex: dict[str, Any]) -> str:
    """Composite key matching data_v2: session/step/state hash, not the (non-unique) id."""
    meta = ex.get("meta") or {}
    state = str(ex.get("state") or "")
    h = hashlib.sha256(state.encode("utf-8")).hexdigest()[:16]
    return f"{meta.get('session', '')}/{meta.get('step', '')}/{h}"


def load_seen(paths: list[str]) -> set[str]:
    seen: set[str] = set()
    for raw in paths:
        p = Path(raw)
        files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            if f.exists():
                for ex in read_jsonl(str(f)):
                    seen.add(decision_key(ex))
    return seen


def load_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.decisions:
        rows = read_jsonl(args.decisions)
        if args.since:
            import datetime

            t0 = datetime.datetime.fromisoformat(args.since).timestamp()
            rows = [r for r in rows if float(r.get("ts") or 0) >= t0]
        examples, summary = from_production(rows)
        summary = {"input": args.decisions, **summary}
    else:
        examples = []
        for f in args.examples:
            examples.extend(read_jsonl(f))
        summary = {"input": args.examples, "n_in": len(examples), "n_out": len(examples)}
    return examples, summary


def predict(engine: Any, examples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    items = [(e["state"], e.get("question") or "", list(e.get("options") or []), e.get("kind", "choice"))
             for e in examples]
    probs = engine._probs(items)
    out: list[dict[str, Any]] = []
    for i, (ex, p) in enumerate(zip(examples, probs)):
        options = list(ex.get("options") or [])
        if not options or len(options) != len(p):
            continue
        target = ex.get("target") or {}
        q = [float(target.get(o, 0.0)) for o in options]
        z = sum(q) or 1.0
        q = [v / z for v in q]
        gold_i = max(range(len(q)), key=lambda j: (q[j], -j))
        pred_i = max(range(len(p)), key=lambda j: (float(p[j]), -j))
        order = sorted(range(len(p)), key=lambda j: -float(p[j]))
        conf = float(p[pred_i])
        second = float(p[order[1]]) if len(order) > 1 else 0.0
        margin = conf - second
        gold = options[gold_i]
        pred = options[pred_i]
        correct = gold_i == pred_i
        reasons: list[str] = []
        if not correct:
            reasons.append("error")
            if conf >= args.high_conf:
                reasons.append("high_conf_error")
        elif margin < args.min_margin:
            reasons.append("low_margin")
        if pred == "shell" and gold != "shell":
            reasons.append("shell_false_positive")
        if gold == "shell" and pred != "shell":
            reasons.append("shell_false_negative")
        priority = priority_of(gold)
        out.append({
            **ex,
            "mine": {
                "gold": gold, "pred": pred, "correct": correct,
                "conf": round(conf, 6), "margin": round(margin, 6),
                "n_options": len(options), "p_top": {options[j]: round(float(p[j]), 6) for j in order[:5]},
                "reasons": reasons, "priority": priority,
                "tool": gold, "needs_review": True,
                "suggested_action": (
                    "inspect label; if gold is correct, collect near-miss negatives"
                    if not correct else "add contrastive near-miss / boundary examples"
                ),
            },
        })
    return out


def summarize(scored: list[dict[str, Any]], queue: list[dict[str, Any]], total: int,
              input_summary: dict[str, Any], seen_skipped: int,
              args: argparse.Namespace) -> dict[str, Any]:
    by_tool: dict[str, Counter[str]] = defaultdict(Counter)
    by_reason: Counter[str] = Counter()
    by_priority: Counter[str] = Counter()
    selected_tool: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scored:
        m = row["mine"]
        by_tool[m["gold"]]["n"] += 1
        by_tool[m["gold"]]["correct"] += int(m["correct"])
        confusion[m["gold"]][m["pred"]] += 1
    for row in queue:
        m = row["mine"]
        by_priority[m["priority"]] += 1
        selected_tool[m["gold"]] += 1
        for r in m["reasons"]:
            by_reason[r] += 1
    tools = {}
    for t, c in sorted(by_tool.items()):
        tools[t] = {"n": c["n"], "correct": c["correct"], "acc": round(c["correct"] / c["n"], 4)}
    return {
        "input_summary": input_summary,
        "n_input_examples": total,
        "n_seen_skipped": seen_skipped,
        "n_queue": len(queue),
        "n_selected_sessions": len({str((r.get("meta") or {}).get("session", "")) for r in queue}),
        "n_selected_by_priority": dict(sorted(by_priority.items())),
        "n_selected_by_reason": dict(sorted(by_reason.items())),
        "n_selected_by_tool": dict(sorted(selected_tool.items())),
        "by_tool": tools,
        "confusion_gold_pred": {g: dict(c.most_common()) for g, c in sorted(confusion.items())},
        "args": vars(args),
    }


def write_summary_md(summary: dict[str, Any], out: Path) -> None:
    lines = [
        "# Hard-case mining queue",
        "",
        f"- input examples: **{summary['n_input_examples']}**",
        f"- already seen (excluded): **{summary['n_seen_skipped']}**",
        f"- selected: **{summary['n_queue']}**",
        f"- selected sessions: **{summary['n_selected_sessions']}**",
        f"- priority counts: `{summary['n_selected_by_priority']}`",
        f"- reason counts: `{summary['n_selected_by_reason']}`",
        f"- selected by tool: `{summary['n_selected_by_tool']}`",
        "",
        "## All scored rows by gold tool",
        "",
        "| tool | n | correct | acc |",
        "|---|---:|---:|---:|",
    ]
    for t, v in summary["by_tool"].items():
        lines.append(f"| {t} | {v['n']} | {v['correct']} | {v['acc']:.2%} |")
    lines += ["", "## Confusion (gold -> predicted)", ""]
    for g, c in summary["confusion_gold_pred"].items():
        lines.append(f"- `{g}` -> {c}")
    lines += [
        "",
        "## How to use",
        "",
        "Review `queue.jsonl`; `mine.reasons` marks errors, low-margin correct calls, and shell bias.",
        "Collect from independent sessions, prefer strong/human labels, and do not add easy shell/read",
        "rows just to increase volume.  Evaluate the next model with per-tool accuracy/confusion.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines))


def select_capped(queue: list[dict[str, Any]], max_per_tool: int = 80, max_total: int = 1000,
                  tool_caps: dict[str, int] | None = None,
                  max_per_session: int = 0) -> list[dict[str, Any]]:
    """Cap a priority-sorted queue.

    ``tool_caps`` (tool -> int) overrides ``max_per_tool`` for the listed tools and is counted
    across all priorities (``0`` excludes the tool, e.g. to mine non-shell cases only).  Other
    tools keep the per-``(priority, gold tool)`` bucket capped by ``max_per_tool``.
    ``max_per_session`` (0 = unlimited) caps rows per ``(session, tool)`` pair so one long
    trajectory cannot dominate a single tool while still contributing to several tools.
    """
    tool_caps = dict(tool_caps or {})
    capped: list[dict[str, Any]] = []
    cap_bucket: Counter[tuple[str, str]] = Counter()
    cap_tool: Counter[str] = Counter()
    cap_session_tool: Counter[tuple[str, str]] = Counter()
    for row in queue:
        m = row["mine"]
        tool = m["gold"]
        if tool in tool_caps:
            if cap_tool[tool] >= tool_caps[tool]:
                continue
        elif cap_bucket[(m["priority"], tool)] >= max_per_tool:
            continue
        session = str((row.get("meta") or {}).get("session", ""))
        if max_per_session and session and cap_session_tool[(session, tool)] >= max_per_session:
            continue
        capped.append(row)
        if tool in tool_caps:
            cap_tool[tool] += 1
        else:
            cap_bucket[(m["priority"], tool)] += 1
        if session:
            cap_session_tool[(session, tool)] += 1
        if len(capped) >= max_total:
            break
    return capped


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--decisions", help="raw production decisions.jsonl")
    src.add_argument("--examples", nargs="+", help="labelled training-format JSONL(s)")
    ap.add_argument("--model", required=True, help="base model path/name")
    ap.add_argument("--adapter", default=None, help="LoRA adapter or full checkpoint")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--exclude", nargs="*", default=[], help="CSV/JSONL files or dirs whose decisions are already seen")
    ap.add_argument("--since", default=None, help="ISO date for --decisions")
    ap.add_argument("--limit", type=int, default=None, help="max input examples")
    ap.add_argument("--max-len", type=int, default=4608)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-margin", type=float, default=0.05,
                    help="select correct decisions with top1-top2 margin below this")
    ap.add_argument("--high-conf", type=float, default=0.8,
                    help="tag errors at/above this confidence")
    ap.add_argument("--max-per-tool", type=int, default=80,
                    help="cap selected rows per (priority, gold tool)")
    ap.add_argument("--tool-caps", default=None,
                    help="JSON {tool: cap} as string or file path; overrides --max-per-tool per "
                         "gold tool (counted across priorities; 0 excludes the tool)")
    ap.add_argument("--max-per-session", type=int, default=0,
                    help="cap selected rows per (session, tool) pair (0 = unlimited)")
    ap.add_argument("--max-total", type=int, default=1000)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    examples, input_summary = load_inputs(args)
    if args.limit:
        examples = examples[: args.limit]
    seen = load_seen(args.exclude)
    n_before_seen = len(examples)
    examples = [e for e in examples if decision_key(e) not in seen]
    seen_skipped = n_before_seen - len(examples)
    print(f"[mine] {len(examples)} examples after seen-filter; loading model...", flush=True)

    from .serve_jev import Engine

    eng = Engine(args.model, args.adapter, args.calibration, name="mine",
                 device=args.device, max_len=args.max_len, batch_size=args.batch_size)
    scored = predict(eng, examples, args)
    queue = [row for row in scored if row["mine"]["reasons"]]

    queue.sort(key=lambda r: (
        PRIORITY_ORDER.get(r["mine"]["priority"], 9),
        0 if "error" in r["mine"]["reasons"] else 1,
        r["mine"]["margin"],
    ))
    tool_caps: dict[str, int] = {}
    if args.tool_caps:
        raw = Path(args.tool_caps)
        tool_caps = json.loads(raw.read_text() if raw.exists() else args.tool_caps)
        bad = [t for t, v in tool_caps.items() if not isinstance(v, int) or v < 0]
        if bad:
            raise SystemExit(f"--tool-caps values must be non-negative ints, got: {bad}")
    queue = select_capped(queue, args.max_per_tool, args.max_total, tool_caps,
                          args.max_per_session)

    with (out / "queue.jsonl").open("w") as f:
        for row in queue:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(scored, queue, len(examples), input_summary, seen_skipped, args)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_summary_md(summary, out)
    print(json.dumps({k: summary[k] for k in ("n_input_examples", "n_queue", "n_selected_by_priority",
                                               "n_selected_by_reason")}, indent=1))
    return summary


if __name__ == "__main__":
    main()