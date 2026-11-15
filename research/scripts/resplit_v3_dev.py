#!/usr/bin/env python3
"""Rebuild the train_v3 split with separate main/alt/shell dev slices.

The old train_v3 dev is dominated by one 38-row alt session and contains no
main-harness shell/edit mass.  This script freezes a new session-level split:

  train       all remaining rows (giant main session and 49ff stay in train)
  dev         main harness only, 11 sessions / 264 rows, primary dev
  dev_alt     alt harness, 22 sessions / 94 rows, secondary dev + rare tools
  dev_shell   9187... / 12 rows, shell_run/browse_web diagnostic only

The original `data/train_v3` is never modified.  Existing checkpoints cannot
be compared on the new dev: these rows were part of the old train, so a fair
comparison requires retraining on `train.jsonl` from this split.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "train_v3"
SRC_TRAIN = SRC_DIR / "train.jsonl"
SRC_DEV = SRC_DIR / "dev.jsonl"
DEFAULT_OUT = ROOT / "data" / "train_v3_clean"

SYNTHETIC = {"grpo-state-check-1", "knockout-check-1", "dedup-check-4"}

# Main-harness holdout.  Eleven sessions, 264 rows; selected to approximate the
# main pool's tool mix and truncation rate while keeping the 975-row session in
# train.  Max session share in dev is 66/264 = 25.0%.
MAIN_DEV = [
    "ses_e27f9491affenDUEuVjrhYiJKj",
    "ses_e35ef9a6effeLpLaZKjKA2iVte",
    "ses_e41d19fefffeLSZ7eP216zX1e1",
    "ses_e41f59941ffeBOrSxRWyCxJLow",
    "ses_e4627c087ffeNJL07HSRCY6EUB",
    "ses_e4722a3e1ffeIp199CyxCVD7m4",
    "ses_e472b82f2ffeqN2rlAy5a5Jhab",
    "ses_e47301e80ffeDQixRbt5yt3H8e",
    "ses_e4732a30cffe8776mo4DuH0Ttg",
    "ses_e6976ef4fffeEk0ykh6w6NTEOR",
    "ses_ed1bb3420ffefOkl5VtEyqvYvm",
]

# Alt-harness holdout.  Twenty-two small sessions / 94 rows, chosen by a
# deterministic distribution-matched search (seed 20261114).  It includes the
# rare web_search / ask_user_question / firecrawl / context7 / present tools;
# shell_run and shell_wait live in 49ff and 9187 and are covered separately.
ALT_DEV = [
    "0ee2e12933b2d6df",
    "1a573751f77bb9a5",
    "1cdb3ca649ea6bd6",
    "298b9b6a49608742",
    "30af184ba1aa9250",
    "3ef2ef649ba277c8",
    "49f54ea123b0c316",
    "5bd872c970d84ca1",
    "5d2df237f0d5a43c",
    "70349e9b3643cb17",
    "780da9018e39b49c",
    "79c4a57d46c73044",
    "7cac5ab1da4142be",
    "89e760c60cb9b3b7",
    "8d67847ebea575d4",
    "9c0f662e0dacfac0",
    "c09a695e6ebe2164",
    "c268a7947b16bae4",
    "cf933a2888029441",
    "d4770d3249b04db7",
    "dc9f1093feb1ef2d",
    "e8cef21f335835ac",
]

SHELL_DEV = ["9187f182bc4a97d5"]

# 49ff (38 shell_run/shell_wait rows) deliberately stays in train.


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (SRC_TRAIN, SRC_DEV):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def session(row: dict[str, Any]) -> str:
    return str(row["meta"]["session"])


def gold(row: dict[str, Any]) -> str:
    target = row.get("target") or {}
    return max(target, key=target.get)


def row_key(row: dict[str, Any]) -> tuple[str, Any, str]:
    """The dedup key used by data_v2: session + step + state hash.

    ``id`` is not unique across decision points (the same step can have
    multiple points); the state hash is what distinguishes them.
    """
    state = str(row.get("state", ""))
    return session(row), (row.get("meta") or {}).get("step"), hashlib.sha256(state.encode()).hexdigest()[:16]


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sessions = collections.Counter(session(r) for r in rows)
    tools = collections.Counter(gold(r) for r in rows)
    options = collections.Counter(len(r.get("options") or []) for r in rows)
    tiers = collections.Counter(str((r.get("meta") or {}).get("label_tier")) for r in rows)
    truncated = sum(bool((r.get("meta") or {}).get("truncated")) for r in rows)
    return {
        "n": len(rows),
        "sessions": len(sessions),
        "max_session_rows": max(sessions.values()) if sessions else 0,
        "max_session_share": round(max(sessions.values()) / len(rows), 4) if rows else 0.0,
        "by_dp": dict(sorted(collections.Counter(r.get("dp") for r in rows).items())),
        "by_kind": dict(sorted(collections.Counter(r.get("kind") for r in rows).items())),
        "by_options": {str(k): v for k, v in sorted(options.items())},
        "by_tool": dict(sorted(tools.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_label_tier": dict(sorted(tiers.items())),
        "truncated": truncated,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: (session(r), (r.get("meta") or {}).get("step") or 0, row_key(r)[2]))
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="replace an existing non-empty output dir")
    args = ap.parse_args()
    out: Path = args.out
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} already exists and is not empty; pass --force to replace it")

    rows = load_rows()
    by_session: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_session[session(row)].append(row)

    all_sessions = set(by_session)
    holdouts = set(MAIN_DEV) | set(ALT_DEV) | set(SHELL_DEV)
    missing = sorted(holdouts - all_sessions)
    if missing:
        raise SystemExit(f"holdout sessions missing from source: {missing}")
    overlap = (set(MAIN_DEV) & set(ALT_DEV)) | (set(MAIN_DEV) & set(SHELL_DEV)) | (set(ALT_DEV) & set(SHELL_DEV))
    if overlap:
        raise SystemExit(f"holdout sets overlap: {sorted(overlap)}")
    if set(SYNTHETIC) & holdouts:
        raise SystemExit("synthetic session is in a holdout set")

    keep = {s for s in all_sessions if s not in SYNTHETIC}
    dev_rows = [r for s in MAIN_DEV for r in by_session[s]]
    alt_rows = [r for s in ALT_DEV for r in by_session[s]]
    shell_rows = [r for s in SHELL_DEV for r in by_session[s]]
    held_rows = dev_rows + alt_rows + shell_rows
    train_rows = [r for s in sorted(keep - holdouts) for r in by_session[s]]

    # Hard checks: no leakage, no duplicate decision rows, exact union.
    split_keys = [{row_key(r) for r in x} for x in (train_rows, dev_rows, alt_rows, shell_rows)]
    if any(split_keys[i] & split_keys[j] for i in range(4) for j in range(i + 1, 4)):
        raise SystemExit("decision row appears in more than one split")
    all_keys = {row_key(r) for r in rows}
    if set().union(*split_keys) != all_keys - {row_key(r) for r in rows if session(r) in SYNTHETIC}:
        raise SystemExit("split union does not cover all non-synthetic rows")
    if len({row_key(r) for r in held_rows}) != len(held_rows):
        raise SystemExit("duplicate decision row within holdouts")

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "dev.jsonl", dev_rows)
    write_jsonl(out / "dev_alt.jsonl", alt_rows)
    write_jsonl(out / "dev_shell.jsonl", shell_rows)

    split_stats = {
        "train": stats(train_rows),
        "dev": stats(dev_rows),
        "dev_alt": stats(alt_rows),
        "dev_shell": stats(shell_rows),
    }
    manifest = {
        "version": "train_v3_clean-v1",
        "source": [str(SRC_TRAIN), str(SRC_DEV)],
        "n_source": len(rows),
        "excluded_synthetic": {s: len(by_session[s]) for s in sorted(set(SYNTHETIC) & all_sessions)},
        "holdout_sessions": {"dev": MAIN_DEV, "dev_alt": ALT_DEV, "dev_shell": SHELL_DEV},
        "splits": split_stats,
        "checks": {
            "session_disjoint": True,
            "row_ids_disjoint": True,
            "union_equals_non_synthetic_source": True,
        },
        "notes": [
            "dev.jsonl is the primary main-harness dev (264 rows).",
            "dev_alt.jsonl is a separate alt-harness / rare-tool slice; do not concatenate it with dev.jsonl for a single aggregate.",
            "dev_shell.jsonl is a 12-row shell_run/browse_web diagnostic, not a representative dev; shell_wait has no held-out rows.",
            "49ff346c6f7a19b5 (38 rows: shell_run 24, shell_wait 7) is in train to improve long-tail coverage.",
            "The 975-row ses_e3fa38decffeZo5PuFEeAC4qaO session stays in train and is never held out.",
            "Existing v2/v3 checkpoints cannot be fairly evaluated on these dev files; retrain on train.jsonl for a clean comparison.",
        ],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    lines = [
        "# train_v3_clean split (v1)",
        "",
        "Rebuilt from `data/train_v3/{train,dev}.jsonl`; the original data is unchanged.",
        "The old dev was 81 rows / 14 sessions, dominated by one 38-row alt session and",
        "with no main-harness shell/edit mass. It also contained 16 rows from v2 train.",
        "",
        "## Files",
        "",
        "| file | rows | sessions | max session share | truncated | dp | purpose |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    purposes = {
        "train": "training (includes giant main session and 49ff)",
        "dev": "primary main-harness dev",
        "dev_alt": "secondary alt-harness + rare-tool dev",
        "dev_shell": "shell_run/browse_web diagnostic only",
    }
    for name in ("train", "dev", "dev_alt", "dev_shell"):
        s = split_stats[name]
        dp = ", ".join(f"{k} {v}" for k, v in s["by_dp"].items())
        trunc = f"{s['truncated'] / s['n']:.1%}"
        lines.append(
            f"| `{name}.jsonl` | {s['n']} | {s['sessions']} | {s['max_session_share']:.1%} | {trunc} | {dp} | {purposes[name]} |"
        )
    lines += [
        "",
        "## Held-out sessions",
        "",
        f"### Main dev ({len(MAIN_DEV)} sessions / {split_stats['dev']['n']} rows)",
        "",
        *[f"- `{s}` ({len(by_session[s])} rows)" for s in MAIN_DEV],
        "",
        f"### Alt dev ({len(ALT_DEV)} sessions / {split_stats['dev_alt']['n']} rows)",
        "",
        *[f"- `{s}` ({len(by_session[s])} rows)" for s in ALT_DEV],
        "",
        "### Shell diagnostic (12 rows)",
        "",
        *[f"- `{s}` ({len(by_session[s])} rows)" for s in SHELL_DEV],
        "",
        "`49ff346c6f7a19b5` (38 rows: shell_run 24, shell_wait 7) is deliberately in train.",
        "The 975-row main session `ses_e3fa38decffeZo5PuFEeAC4qaO` is deliberately in train.",
        "",
        "## Evaluation rules",
        "",
        "- Use `dev.jsonl` for the primary model-selection metric.",
        "- Report `dev_alt.jsonl` and `dev_shell.jsonl` separately; do not pool them into one accuracy.",
        "- The old v2/v3 checkpoints saw most of these rows during training. Any comparison on this split",
        "  requires retraining on `train.jsonl` (or using only the old 65-row clean subset with a caveat).",
        "- Prefer per-tool accuracy / confusion in addition to aggregate accuracy; the aggregate hides",
        "  the shell_run / final_result / stop behavior. shell_wait has no held-out rows.",
        f"- `dev.jsonl` has {split_stats['dev']['truncated']}/{split_stats['dev']['n']} truncated rows "
        f"({split_stats['dev']['truncated'] / split_stats['dev']['n']:.1%}); the full main pool is "
        f"{sum(bool((r.get('meta') or {}).get('truncated')) for r in rows if session(r).startswith('ses_') and session(r) not in SYNTHETIC)}"
        f"/{sum(1 for r in rows if session(r).startswith('ses_') and session(r) not in SYNTHETIC)}. "
        "Report truncated / non-truncated slices if that difference matters for the decision.",
        "",
        "## Tool distribution",
        "",
        "| split | tools |",
        "|---|---|",
    ]
    for name in ("train", "dev", "dev_alt", "dev_shell"):
        tools = ", ".join(f"{k} {v}" for k, v in split_stats[name]["by_tool"].items())
        lines.append(f"| `{name}` | {tools} |")
    lines += [
        "",
        "## Rebuild",
        "",
        "```bash",
        "cd research",
        ".venv/bin/python scripts/resplit_v3_dev.py --force",
        "```",
        "",
    ]
    (out / "SPLIT.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"out": str(out), "splits": {k: v["n"] for k, v in split_stats.items()}}, indent=1))


if __name__ == "__main__":
    main()