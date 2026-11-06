"""Era filters for the log readers: keep the rows a decision model produced between a date, with (or without)
the calibration temperature and the state truncation applied. The rows carry these fields themselves —
``decisions.jsonl`` at the top level, ``steps.jsonl`` under ``decision`` — so the filter is the same predicate
for both; ``jah report``/``jah fit`` expose them as --since/--decision-model/--calibrated/--truncated so the
pooled numbers stop mixing distributions that no longer exist."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, Iterable

SINCE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")
TRUE = ("1", "true", "yes", "on")
FALSE = ("0", "false", "no", "off")


def parse_since(value: str) -> float:
    """A local-time lower bound for --since: 'YYYY-MM-DD[ HH:MM[:SS]]' or an epoch number."""
    v = value.strip()
    try:
        return float(v)
    except ValueError:
        pass
    for fmt in SINCE_FORMATS:
        try:
            return datetime.strptime(v, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"cannot read a time from {value!r}; use 'YYYY-MM-DD[ HH:MM[:SS]]' or an epoch number")


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise ValueError(f"expected true/false (also 1/0, yes/no, on/off), got {value!r}")


def _field(r: dict[str, Any], key: str) -> Any:
    v = r.get(key)
    if v is None:
        d = r.get("decision")
        if isinstance(d, dict):
            v = d.get(key)
    return v


@dataclasses.dataclass(frozen=True)
class Era:
    """The rows a filter keeps. An unset part matches everything; rows from before the fields existed count as
    calibrated=False/truncated=False (they were produced without them)."""
    since: float | None = None
    model: str | None = None          # case-insensitive substring of the decision model name
    calibrated: bool | None = None
    truncated: bool | None = None

    def keep(self, r: dict[str, Any]) -> bool:
        if self.since is not None and float(r.get("ts") or 0) < self.since:
            return False
        if self.model is not None and self.model.lower() not in str(_field(r, "model") or "").lower():
            return False
        if self.calibrated is not None and bool(_field(r, "calibrated")) != self.calibrated:
            return False
        if self.truncated is not None and bool(_field(r, "truncated")) != self.truncated:
            return False
        return True

    def note(self) -> str:
        """A human-readable summary of what is active ('' when nothing is)."""
        parts = []
        if self.since is not None:
            parts.append("since " + datetime.fromtimestamp(self.since).strftime("%Y-%m-%d %H:%M"))
        if self.model:
            parts.append(f"decision model ~{self.model!r}")
        for name in ("calibrated", "truncated"):
            v = getattr(self, name)
            if v is not None:
                parts.append(f"{name}={str(v).lower()}")
        return "; ".join(parts)


def apply(rows: Iterable[dict[str, Any]], era: Era) -> list[dict[str, Any]]:
    """Filter decision rows (or plain step rows) by the era."""
    if not era.note():
        return list(rows)
    return [r for r in rows if era.keep(r)]


def apply_steps(rows: Iterable[dict[str, Any]], era: Era) -> list[dict[str, Any]]:
    """Filter steps.jsonl rows, keeping cache-accounting rows attached to surviving sessions (they carry no
    decision fields of their own)."""
    rows = list(rows)
    if not era.note():
        return rows
    steps = [r for r in rows if r.get("kind") != "cache"]
    kept = [r for r in steps if era.keep(r)]
    sessions = {r.get("session") for r in kept}
    kept += [r for r in rows if r.get("kind") == "cache" and r.get("session") in sessions
             and (era.since is None or float(r.get("ts") or 0) >= era.since)]
    return kept


def from_args(a: Any) -> Era:
    return Era(since=getattr(a, "since", None), model=getattr(a, "decision_model", None),
               calibrated=getattr(a, "calibrated", None), truncated=getattr(a, "truncated", None))
