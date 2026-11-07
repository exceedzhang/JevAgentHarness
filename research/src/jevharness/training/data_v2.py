"""Training data v2 (production): decision examples harvested from a deployed proxy's ``decisions.jsonl``.

The standalone proxy (``litellm_jah``) logs one row per decision point per step
(``point``/``mode``/``question``/``options``/``p``/``choice``/``label``/``fit_ok``/...). With
``log_decision_state: true`` each successful row also carries the exact ``state`` string that was
sent to the decision model (post-truncation), which makes the row a complete training prompt:
``state`` + ``question`` + ``options`` is rendered by ``data.render`` byte-identically to what the
served model sees.

Labels: the production log's ``label`` is the System-2 ground truth at that state - for ``tool_select``
the tool the agent actually called (label_tier fast/strong), for ``stop`` the verdict of the strong
finish-check redo (shadow mode), for ``routing`` there is no label (no redo) and those rows are
skipped unless ``--include-routing`` asks for them with the logged model's own p as a soft target.

Only rows with ``fit_ok`` (the label was produced without the decision model's influence - the same
guard ``jah fit`` uses) are usable as gold; rows are deduped per (session, step, point) because a
client retry of one step writes only one labelled decision (the settle layer guarantees it).

Usage::

    python -m jevharness.training.data_v2 ~/.jah/litellm-logs/decisions.jsonl \
        --out data/train_v2/prod.jsonl [--dev-frac 0.1] [--since 2026-11-07]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .data import OPTION_LABELS, _onehot, cap_options, read_jsonl, split_by_task, stats, write_jsonl

#: decision points that carry a System-2 gold label in the production log
LABELLED_POINTS = ("tool_select", "stop")


def from_production(rows: Iterable[dict[str, Any]], min_options: int = 2,
                    max_options: int = len(OPTION_LABELS), seed: int = 0,
                    include_routing: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert production decision rows to training examples (data.py format).

    Returns (examples, summary): the summary records every skip reason, so a silent loss of data
    (a tool rename, an unpriced era, a missing state) is visible in the manifest rather than in a
    smaller-than-expected file.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    skip: Counter[str] = Counter()
    by_point: Counter[str] = Counter()
    n_in = 0
    for r in rows:
        n_in += 1
        point = r.get("point")
        if point not in LABELLED_POINTS and not (include_routing and point == "routing"):
            skip["point_unlabelled" if point == "routing" else f"point:{point}"] += 1
            continue
        if r.get("error"):
            skip["error_row"] += 1
            continue
        state = r.get("state")
        if not state:
            skip["no_state"] += 1
            continue
        opts = [str(o) for o in (r.get("options") or [])]
        if len(opts) < min_options:
            skip["too_few_options"] += 1
            continue
        label = r.get("label")
        if point == "routing":  # no redo label: the model's own distribution is the target (soft)
            p = {o: float(v) for o, v in (r.get("p") or {}).items() if float(v) > 0}
            if not p or sum(p.values()) <= 0:
                skip["routing_no_p"] += 1
                continue
            target = p  # already normalized by the backend
            meta_extra = {"label_source": "model_soft"}
        else:
            if label is None or not r.get("fit_ok"):
                skip["unlabelled_or_not_fit_ok"] += 1
                continue
            label = str(label)
            if label not in opts:
                # the agent's tool may have been renamed between the decision and the settle (the log
                # keeps the name the agent used); a stale label is not a learning signal
                skip["label_not_in_options"] += 1
                continue
            target = _onehot(opts, label)
            meta_extra = {"label_source": "s2", "label_tier": r.get("label_tier")}
        eid_base = f"prod:{r.get('session')}/{r.get('step')}/{point}"
        if eid_base in seen:  # one (session, step, point) is one sample, same as `jah fit`
            skip["duplicate"] += 1
            continue
        seen.add(eid_base)
        ex_opts = list(opts)
        if len(ex_opts) > max_options:
            ex_opts = cap_options(ex_opts, label, max_options, random.Random(f"{seed}:{eid_base}"))
            meta_extra["n_options_logged"] = len(opts)
        meta = {**meta_extra, "session": r.get("session"), "step": r.get("step"), "ts": r.get("ts"),
                "model": r.get("model"), "calibrated": r.get("calibrated"), "truncated": r.get("truncated"),
                "logged": {"choice": r.get("choice"), "p_top": {k: round(float(v), 4)
                for k, v in sorted((r.get("p") or {}).items(), key=lambda kv: -float(kv[1]))[:5]}}}
        out.append({"id": eid_base, "source": "production", "dp": point, "kind": "choice",
                    "state": state, "question": r.get("question") or "", "options": ex_opts,
                    "target": target, "q_values": None, "meta": meta})
        by_point[point] += 1
    summary = {"n_in": n_in, "n_out": len(out), "skipped": dict(skip), "by_point": dict(by_point)}
    return out, summary


def build(path: str | Path, dev_frac: float = 0.1, seed: int = 0, out_dir: str | Path | None = None,
          since: str | None = None) -> dict[str, Any]:
    """Read a ``decisions.jsonl``, write ``prod.jsonl`` / ``prod_dev.jsonl`` (task-split by session) and a
    ``MANIFEST.json`` with the skip reasons. Returns the manifest dict."""
    rows = read_jsonl(path)
    if since:
        import datetime

        t0 = datetime.datetime.fromisoformat(since).timestamp()
        rows = [r for r in rows if float(r.get("ts") or 0) >= t0]
    examples, summary = from_production(rows, seed=seed)
    if out_dir is None:
        out_dir = Path("data/train_v2")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(examples) < 2 or dev_frac <= 0:
        train, dev = examples, []
    else:
        train, dev = split_by_task(examples, dev_frac=dev_frac, seed=seed)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(dev, out_dir / "dev.jsonl")
    manifest = {"built": str(Path(path)), "seed": seed, "dev_frac": dev_frac, **summary,
                "stats_train": stats(train), "stats_dev": stats(dev)}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("decisions", help="path to decisions.jsonl (the proxy journal)")
    ap.add_argument("--out", default="data/train_v2", help="output directory (default data/train_v2)")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--since", default=None, help="ISO date; keep rows from this moment on")
    a = ap.parse_args(argv)
    m = build(a.decisions, dev_frac=a.dev_frac, seed=a.seed, out_dir=a.out, since=a.since)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
