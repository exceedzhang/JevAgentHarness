#!/usr/bin/env python3
"""Build a weighted hard-case training set from the mined review queue.

The queue is a *review* artifact: it contains rows from train/dev/dev_alt/
dev_shell and may contain duplicate or truncated states.  For training we:

1. keep only queue rows that belong to ``train_v3_clean/train.jsonl`` or are
   genuinely unseen (not in any clean split);
2. drop dev/dev_alt/dev_shell rows and truncated states;
3. cap by session and gold tool so one trajectory or tool cannot dominate;
4. oversample P0/P1/P2 hard cases while leaving P3 at 1x;
5. append the extra copies to the clean train file.  ``build_rows`` does not
   deduplicate, so repeated JSONL rows are intentional.

Run from ``research/``::

    .venv/bin/python scripts/build_hard_weighted.py

Outputs (defaults):
  data/train_v3_clean/train_hard_weighted_v1.jsonl
  data/train_v3_clean/train_hard_cases_v1.jsonl   # selected rows, audits only
  data/train_v3_clean/MANIFEST_hard_weighted_v1.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


TOOL_CAPS = {
    "shell": 20,
    "read": 40,
    "edit": 40,
    "reply_without_tool": 40,
    "compress": 40,
    "grep": 40,
    "write": 40,
    "glob": 18,
    "webfetch": 16,
    "execute": 16,
    "not_done": 15,
    "bash": 14,
    "shell_run": 8,
    "shell_wait": 6,
    "done": 5,
    "question": 7,
    "skill": 5,
    "context7_resolve-library-id": 4,
    "context7_query-docs": 3,
    "websearch": 4,
    "web_fetch": 2,
    "firecrawl-mcp_firecrawl_search": 2,
    "web_search": 2,
    "browse_web": 2,
    "ask_user_question": 1,
    "ask_human": 1,
    "present": 1,
    "final_result": 1,
}
SESSION_CAP = 50
PRIORITY_WEIGHT = {"P0": 3, "P1": 2, "P2": 2, "P3": 1}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def key(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    session = str(meta.get("session", ""))
    step = str(meta.get("step", ""))
    digest = hashlib.sha256(str(row.get("state", "")).encode()).hexdigest()[:16]
    return f"{session}\x00{step}\x00{digest}"


def gold_tool(row: dict[str, Any]) -> str:
    mine = row.get("mine") or {}
    if mine.get("tool"):
        return str(mine["tool"])
    target = row.get("target") or {}
    return max(target, key=target.get)


def priority(row: dict[str, Any]) -> str:
    return str((row.get("mine") or {}).get("priority", "P3"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-dir", type=Path, default=Path("data/train_v3_clean"))
    ap.add_argument("--queue", type=Path, default=Path("checkpoints/mine_hard_cases_p1_all/queue.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/train_v3_clean/train_hard_weighted_v1.jsonl"))
    ap.add_argument("--selected-out", type=Path, default=Path("data/train_v3_clean/train_hard_cases_v1.jsonl"))
    ap.add_argument("--manifest-out", type=Path, default=Path("data/train_v3_clean/MANIFEST_hard_weighted_v1.json"))
    ap.add_argument("--version", default="v1", help="manifest version suffix, e.g. v2")
    ap.add_argument("--session-cap", type=int, default=None, help="override SESSION_CAP")
    ap.add_argument("--tool-caps", default=None,
                    help="JSON {tool: cap} as string or file path; merged over the built-in caps")
    ap.add_argument("--priority-weights", default=None,
                    help='JSON {"P0": 3, ...}; merged over PRIORITY_WEIGHT')
    args = ap.parse_args()
    session_cap = SESSION_CAP if args.session_cap is None else args.session_cap
    tool_caps = dict(TOOL_CAPS)
    if args.tool_caps:
        p = Path(args.tool_caps)
        tool_caps.update(json.loads(p.read_text() if p.exists() else args.tool_caps))
    weights = dict(PRIORITY_WEIGHT)
    if args.priority_weights:
        p = Path(args.priority_weights)
        weights.update({str(k): int(v) for k, v in
                        json.loads(p.read_text() if p.exists() else args.priority_weights).items()})

    clean = read_jsonl(args.clean_dir / "train.jsonl")
    queue = read_jsonl(args.queue)
    split_rows = {
        name: read_jsonl(args.clean_dir / f"{name}.jsonl")
        for name in ("train", "dev", "dev_alt", "dev_shell")
    }
    split_keys = {name: {key(r) for r in rows} for name, rows in split_rows.items()}
    dev_keys = set().union(split_keys["dev"], split_keys["dev_alt"], split_keys["dev_shell"])
    train_keys = split_keys["train"]

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = collections.Counter()
    by_session: collections.Counter[str] = collections.Counter()
    by_tool: collections.Counter[str] = collections.Counter()

    for row in queue:
        k = key(row)
        if k in seen:
            skipped["duplicate_queue"] += 1
            continue
        seen.add(k)
        if k in dev_keys:
            skipped["dev_or_leak"] += 1
            continue
        if bool((row.get("meta") or {}).get("truncated")):
            skipped["truncated"] += 1
            continue
        session = str((row.get("meta") or {}).get("session", ""))
        tool = gold_tool(row)
        cap = tool_caps.get(tool, 1)
        if by_session[session] >= session_cap:
            skipped["session_cap"] += 1
            continue
        if by_tool[tool] >= cap:
            skipped["tool_cap"] += 1
            continue
        selected.append(row)
        by_session[session] += 1
        by_tool[tool] += 1

    # Repeated rows are intentional: build_rows does not deduplicate.
    extras: list[dict[str, Any]] = []
    for row in selected:
        weight = weights.get(priority(row), 1)
        copies = weight if key(row) not in train_keys else max(0, weight - 1)
        extras.extend([row] * copies)

    out = clean + extras
    selected_by_priority = collections.Counter(priority(r) for r in selected)
    weighted_hard_by_tool = collections.Counter(gold_tool(r) for r in extras)
    final_by_tool = collections.Counter(gold_tool(r) for r in out)
    # Hard guarantee: no held-out row can enter training.
    bad = [r for r in out if key(r) in dev_keys]
    if bad:
        raise SystemExit(f"refusing to write: {len(bad)} dev rows leaked into train")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, out)
    write_jsonl(args.selected_out, selected)

    manifest = {
        "version": f"train_hard_weighted_{args.version}",
        "clean_train": str(args.clean_dir / "train.jsonl"),
        "queue": str(args.queue),
        "out": str(args.out),
        "selected_out": str(args.selected_out),
        "caps": {"session": session_cap, "tool": tool_caps},
        "priority_weight": weights,
        "queue_rows": len(queue),
        "queue_unique_keys": len(seen),
        "selected_hard_cases": len(selected),
        "selected_sessions": len(by_session),
        "selected_by_tool": dict(sorted(by_tool.items())),
        "selected_by_session_top": by_session.most_common(10),
        "extras": len(extras),
        "clean_rows": len(clean),
        "final_rows": len(out),
        "skipped": dict(sorted(skipped.items())),
        "notes": [
            "Queue rows already present in clean train are not duplicated as base rows; only weight-1 copies are appended.",
            "Unseen valid queue rows are appended weight copies.",
            "Dev/dev_alt/dev_shell rows are never appended.",
            "Truncated queue rows are excluded.",
        ],
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in (
        "queue_rows", "queue_unique_keys", "selected_hard_cases", "selected_sessions",
        "extras", "clean_rows", "final_rows", "skipped",
    )}, ensure_ascii=False, indent=1))
    print("selected_by_tool:", json.dumps(manifest["selected_by_tool"], ensure_ascii=False))
    print("selected_by_session_top:", manifest["selected_by_session_top"])


if __name__ == "__main__":
    main()