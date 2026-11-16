#!/usr/bin/env python3
"""Mine boundary hard cases: adjacent decisions in one session whose gold tools are two
different members of a confusion family (default: read / reply_without_tool / edit).

The selection is *structural* and needs no model scoring: within a session, two decisions
whose steps are close and whose gold tools alternate inside the family differ only by the
smallest state increment, so training both rows as a weighted pair targets the exact
read<->reply / edit<->read confusion that per-tool mining (v5) showed pure volume cannot fix.

Row shape: the standard data_v2 example (``state``/``question``/``options``/``target``/
``meta{session,step}``), with a ``mine`` block added.  Fresh production rows (``--decisions``)
that are family-gold but not in a pair are included as P1 singletons.

Run from ``research/``::

    .venv/bin/python scripts/mine_boundary_pairs.py \
        --examples data/train_v4/train.jsonl \
        --decisions data/decisions.jsonl \
        --exclude data/train_v4/train.jsonl data/train_v4/dev.jsonl data/train_v4/dev_alt.jsonl \
                  data/train_v4/dev_shell.jsonl data/train_v3_clean/train_hard_cases_v1.jsonl \
                  data/train_v3_clean/train_hard_cases_v2.jsonl data/train_v3_clean/train_hard_cases_v3.jsonl \
                  data/train_v5/train_hard_cases_v5.jsonl checkpoints/queue_v5_merged.jsonl \
        --out checkpoints/mine_boundary_v6

The output queue is consumed by ``scripts/build_hard_weighted.py`` (``--base``/``--dev-dir``).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jevharness.training.data_v2 import from_production  # noqa: E402
from jevharness.training.mine_hard_cases import (  # noqa: E402
    TOOL_ALIASES,
    decision_key,
    load_seen,
    read_jsonl,
    tool_of,
)

FAMILY_DEFAULT = ["read", "reply_without_tool", "edit"]
DEFAULT_CAPS = {"read": 100, "reply_without_tool": 80, "edit": 80}


def write_jsonl(path: Path, rows: Any) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def canon(tool: str) -> str:
    return TOOL_ALIASES.get(tool, tool)


def select_boundary_pairs(rows: list[dict[str, Any]], family: set[str],
                          max_gap: int) -> tuple[set[int], list[tuple[int, int, str]]]:
    """Return (indices of rows that are pair members, list of (i, j, pair_id)).

    A pair is two rows of the same session whose gold tools are two different family members
    and whose steps differ by at most ``max_gap`` (they are then adjacent in the sorted
    session sequence; a gap of 2 tolerates one skipped/stop step between them).
    """
    by_session: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for idx, row in enumerate(rows):
        meta = row.get("meta") or {}
        tool = canon(tool_of(row))
        if tool in family:
            try:
                step = int(meta.get("step", 0))
            except (TypeError, ValueError):
                step = 0
            by_session[str(meta.get("session", ""))].append((step, idx))
    members: set[int] = set()
    pairs: list[tuple[int, int, str]] = []
    for session, steps in by_session.items():
        steps.sort()
        for a, b in zip(steps, steps[1:]):
            if b[0] - a[0] > max_gap:
                continue
            # find golds via rows for the two original indices
            i, j = a[1], b[1]
            ti, tj = canon(tool_of(rows[i])), canon(tool_of(rows[j]))
            if ti != tj:
                pairs.append((i, j, f"{session}@{a[0]}"))
                members.add(i)
                members.add(j)
    return members, pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--examples", nargs="*", default=[], help="labelled pool JSONL files (data_v2 format)")
    ap.add_argument("--decisions", default=None, help="production decisions.jsonl for fresh rows")
    ap.add_argument("--since", default=None, help="ISO timestamp: only decisions at/after this time")
    ap.add_argument("--exclude", nargs="*", default=[], help="files/dirs of already-used rows (leak guard)")
    ap.add_argument("--family", default=",".join(FAMILY_DEFAULT),
                    help="comma list of canonical family tools (default: read,reply_without_tool,edit)")
    ap.add_argument("--max-gap", type=int, default=2, help="max step difference for a pair (2 = one step skipped)")
    ap.add_argument("--tool-caps", default=None,
                    help="JSON {canonical tool: cap} as string or file (default: read 100, reply 80, edit 80)")
    ap.add_argument("--max-per-session-tool", type=int, default=4,
                    help="per (session, canonical tool) cap on queue rows")
    ap.add_argument("--max-total", type=int, default=400)
    ap.add_argument("--no-prod-singletons", action="store_true",
                    help="do not include fresh production family rows that are not in a pair")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/mine_boundary_v6"))
    args = ap.parse_args()

    family = {canon(t.strip()) for t in args.family.split(",") if t.strip()}
    caps = dict(DEFAULT_CAPS)
    if args.tool_caps:
        p = Path(args.tool_caps)
        caps.update({str(k): int(v) for k, v in
                     json.loads(p.read_text() if p.exists() else args.tool_caps).items()})
    bad = [t for t, v in caps.items() if not isinstance(v, int) or v < 0]
    if bad:
        raise SystemExit(f"--tool-caps values must be non-negative ints, got: {bad}")

    # --- pool rows -----------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for f in args.examples:
        rows.extend(read_jsonl(f))
    for r in rows:
        r.setdefault("meta", {})
    seen = load_seen(args.exclude)
    seen |= {decision_key(r) for r in rows}  # pool rows are themselves excluded from prod
    pool_keys = {decision_key(r) for r in rows}
    n_pool = len(rows)

    # --- fresh production rows ------------------------------------------------
    n_prod_new = 0
    prod_summary: dict[str, Any] = {}
    if args.decisions:
        raw = read_jsonl(args.decisions)
        if args.since:
            import datetime
            t0 = datetime.datetime.fromisoformat(args.since).timestamp()
            raw = [r for r in raw if float(r.get("ts") or 0) >= t0]
        examples, prod_summary = from_production(raw)
        fresh = []
        for ex in examples:
            if decision_key(ex) in seen:
                continue
            fresh.append(ex)
        seen |= {decision_key(e) for e in fresh}
        n_prod_new = len(fresh)
        rows.extend(fresh)
    for r in rows[n_pool:]:
        r.setdefault("meta", {})

    # --- boundary selection ----------------------------------------------------
    members, pairs = select_boundary_pairs(rows, family, args.max_gap)
    n_pairs = len(pairs)
    pair_of = {}
    for i, j, pid in pairs:
        pair_of.setdefault(i, []).append((j, pid))
        pair_of.setdefault(j, []).append((i, pid))

    queue: list[dict[str, Any]] = []
    skipped = collections.Counter()
    by_tool = collections.Counter()
    by_session_tool = collections.Counter()
    for idx, row in enumerate(rows):
        tool = canon(tool_of(row))
        is_pair = idx in members
        is_prod = idx >= n_pool
        if not is_pair and is_prod and not args.no_prod_singletons:
            if tool not in family:
                continue
            pri, reasons = "P1", ["boundary_prod_new"]
        elif is_pair:
            pri, reasons = "P0", ["boundary_pair"]
        else:
            continue
        if by_tool[tool] >= caps.get(tool, 0):
            skipped["tool_cap"] += 1
            continue
        session = str(row["meta"].get("session", ""))
        st = (session, tool)
        if by_session_tool[st] >= args.max_per_session_tool:
            skipped["session_tool_cap"] += 1
            continue
        mine = {"priority": pri, "gold": tool_of(row), "tool": tool, "reasons": reasons}
        if is_pair:
            mine["pair_id"] = pair_of[idx][0][1]
            mine["pair_partners"] = [q for q, _pid in pair_of[idx][:2]]
        out = dict(row)
        out["mine"] = mine
        queue.append(out)
        by_tool[tool] += 1
        by_session_tool[st] += 1
        if len(queue) >= args.max_total:
            break

    # P0 first, stable within priority (input order preserved)
    queue.sort(key=lambda r: (r["mine"]["priority"] != "P0"))

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "queue.jsonl", queue)
    selected_by_reason = collections.Counter(r["mine"]["reasons"][0] for r in queue)
    selected_by_session = collections.Counter(str(r["meta"].get("session", "")) for r in queue)
    summary = {
        "pool_rows": n_pool,
        "prod_fresh_rows": n_prod_new,
        "prod_skipped": prod_summary.get("skipped", {}),
        "n_pairs": n_pairs,
        "pair_member_rows": len(members),
        "queue_rows": len(queue),
        "queue_sessions": len(selected_by_session),
        "selected_by_tool": dict(sorted(by_tool.items())),
        "selected_by_reason": dict(sorted(selected_by_reason.items())),
        "skipped": dict(sorted(skipped.items())),
        "family": sorted(family),
        "caps": caps,
        "max_gap": args.max_gap,
        "max_per_session_tool": args.max_per_session_tool,
        "excluded": args.exclude,
        "notes": [
            "Pairs are structural (same session, close steps, gold alternates inside the family); no model scoring.",
            "Fresh production family rows outside pairs are P1 singletons unless --no-prod-singletons.",
            "Caps are on canonical tool names; per (session, canonical tool) capped separately.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = ["# Boundary pair mining", ""]
    lines += [f"- pool rows: **{n_pool}**, prod fresh: **{n_prod_new}**",
              f"- pairs: **{n_pairs}**, pair-member rows: **{len(members)}**",
              f"- queue: **{len(queue)}** rows / {len(selected_by_session)} sessions",
              f"- by tool: {json.dumps(summary['selected_by_tool'])}", ""]
    (args.out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("pool_rows", "prod_fresh_rows", "n_pairs", "queue_rows", "selected_by_tool", "skipped")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
