#!/usr/bin/env python3
"""Resplit the clean pool into train_v4 with a larger, stratified dev (11-15).

The v3 clean split (train 2189 / dev 264 / dev_alt 94 / dev_shell 12) held out only
264 dev rows; weak-tool coverage was poor (compress 10, grep 7, glob 6). This script
builds ``data/train_v4/`` with a ~3.5x larger dev while keeping every v3 result
reproducible (the old directory is left untouched).

Rules (approved plan):
1. pool = the four train_v3_clean files (disjoint), deduped by
   (session, step, sha256(state)[:16]);
2. the top-3 giant sessions (60.5% of all rows) go to train only - one of them in
   dev would dominate the metric;
3. the legacy holdout sessions (dev + dev_alt + dev_shell) stay entirely in dev,
   so v1-v3 per-tool numbers remain comparable;
4. further dev sessions are chosen greedily (whole sessions, zero leak) to fill the
   per-gold-tool quota; tools the pool cannot reach >= 30 are taken in full;
5. new train = everything else.

Outputs (defaults; ``--dry-run`` prints the plan and writes nothing):
  data/train_v4/train.jsonl      data/train_v4/dev.jsonl
  data/train_v4/dev_alt.jsonl    dev rows that are NOT legacy (new additions)
  data/train_v4/dev_shell.jsonl  dev rows whose gold is a shell tool
  data/train_v4/MANIFEST_v4.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

# dev target per gold tool; the pool cannot reach 30 for the weak ones, so they are
# "take everything available" quotas (28/22/16/20/25/15), not a promise of >= 30.
QUOTA = {
    "read": 120,
    "edit": 90,
    "reply_without_tool": 120,
    "shell": 150,
    "done": 30,
    "final_result": 71,
    "compress": 28,
    "grep": 22,
    "glob": 16,
    "write": 20,
    "webfetch": 25,
    "not_done": 15,
}
# every other gold tool (web_search, execute, bash, skill, ...): up to 10 per tool
OTHER_CAP = 10
GIANT_N = 3


def session(row: dict) -> str:
    meta = row.get("meta") or {}
    return meta.get("session") or session(row) or "?"


def key(row: dict) -> tuple:
    meta = row.get("meta") or {}
    state = row.get("state") or ""
    return (session(row), meta.get("step", row.get("step")),
            hashlib.sha256(state.encode("utf-8")).hexdigest()[:16])


def gold(row: dict) -> str:
    target = row.get("target") or {}
    best, best_v = None, -1
    for opt, v in target.items():
        if v > best_v:
            best, best_v = opt, v
    return best or "?"


def load_pool(clean_dir: Path) -> tuple[dict, set]:
    """Dedup the four clean files; return {key: row} and the legacy (holdout) sessions."""
    pool: dict = {}
    legacy_sessions: set = set()
    for name in ("train.jsonl", "dev.jsonl", "dev_alt.jsonl", "dev_shell.jsonl"):
        p = clean_dir / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if name != "train.jsonl":
                legacy_sessions.add(session(row))
            k = key(row)
            pool.setdefault(k, row)
    return pool, legacy_sessions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", type=Path, default=Path("data/train_v3_clean"))
    ap.add_argument("--out", type=Path, default=Path("data/train_v4"))
    ap.add_argument("--dev-session-cap", type=int, default=50,
                    help="max dev sessions (whole-session split, so more sessions = more train sessions lost)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pool, legacy_sessions = load_pool(args.clean_dir)
    by_session: dict = collections.defaultdict(list)
    for k, row in pool.items():
        by_session[session(row)].append(row)
    ranked = sorted(by_session.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    giants = {s for s, _ in ranked[:GIANT_N]}

    dev_rows: dict = {}
    # 1) legacy holdout sessions stay in dev in full
    for s in legacy_sessions:
        if s in by_session:
            for row in by_session[s]:
                dev_rows[key(row)] = row

    counts = collections.Counter(gold(r) for r in dev_rows.values())
    remaining = {t: max(0, QUOTA[t] - counts[t]) for t in QUOTA}
    other_remaining = collections.defaultdict(lambda: OTHER_CAP)

    # 2) greedy: whole sessions (largest first), weighted by how far each tool is
    # from its quota; stop at the dev-session cap so train keeps enough sessions
    dev_session_set = set(legacy_sessions) & set(by_session)
    for s, rows in ranked:
        if len(dev_session_set) >= args.dev_session_cap:
            break
        if s in giants or s in legacy_sessions or s in dev_session_set:
            continue
        tool_n = collections.Counter(gold(r) for r in rows)
        score = sum(min(n, remaining[t]) / max(1, QUOTA[t]) for t, n in tool_n.items() if t in remaining)
        score += sum(min(n, other_remaining[t]) for t, n in tool_n.items() if t not in remaining) / OTHER_CAP
        if score <= 0:
            continue
        for row in rows:
            dev_rows[key(row)] = row
        dev_session_set.add(s)
        for t, n in tool_n.items():
            if t in remaining:
                remaining[t] = max(0, remaining[t] - n)
        if all(v == 0 for v in remaining.values()):
            break

    dev = list(dev_rows.values())
    dev_sessions = {session(r) for r in dev}
    train = [r for r in pool.values() if session(r) not in dev_sessions]
    # sanity: no session on both sides; giants in train
    assert not (dev_sessions & {session(r) for r in train}), "session leak between train and dev"
    for g in giants:
        assert g in {session(r) for r in train}, "giant session missing from train"

    dev_shell = [r for r in dev if "shell" in (gold(r) or "").lower()]
    dev_alt = [r for r in dev if session(r) not in legacy_sessions]

    def table(rows: list) -> dict:
        return dict(collections.Counter(gold(r) for r in rows))

    plan = {
        "pool": len(pool),
        "sessions": len(by_session),
        "giants": {s: len(by_session[s]) for s in giants},
        "dev": len(dev),
        "dev_sessions": len(dev_sessions),
        "dev_legacy_rows": sum(1 for r in dev if session(r) in legacy_sessions),
        "dev_new_rows": len(dev_alt),
        "dev_shell_rows": len(dev_shell),
        "train": len(train),
        "quota_check": {t: {"dev": table(dev).get(t, 0), "quota": QUOTA[t]} for t in QUOTA},
        "train_by_tool": table(train),
        "dev_by_tool": table(dev),
        "top_dev_sessions": {s: n for s, n in collections.Counter(session(r) for r in dev).most_common(8)},
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("dry-run: nothing written")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("dev.jsonl", dev),
                       ("dev_alt.jsonl", dev_alt), ("dev_shell.jsonl", dev_shell)):
        (args.out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (args.out / "MANIFEST_v4.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}/ (train {len(train)}, dev {len(dev)}, alt {len(dev_alt)}, shell {len(dev_shell)})")


if __name__ == "__main__":
    main()
