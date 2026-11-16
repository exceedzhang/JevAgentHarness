"""``jah fit``: thresholds for the decision points, fitted on your own logged runs.

The paper fits each decision point's threshold on trajectories in which the decision model answered every question
silently (shadow mode), so that its answer is used only where it is reliable, and the LLM decides below the
threshold. ``jah fit`` does a simplified version of that fit with the proxy's logs (``decisions.jsonl``). The target is
agreement with the LLM: a decision is *labelled* when the LLM's own choice at that point is known and was made without
the decision model's influence (``fit_ok``). For each point, the fitted tau is the smallest threshold at which the
decisions the point would act on agree with the LLM at the target rate (default 95%), counted on at least
``min_count`` decisions and judged by a one-sided Wilson lower bound (``z``), so that a handful of lucky agreements
does not pass. If no threshold qualifies, the LLM should keep that decision (mode ``off`` or ``shadow``).

``tool_select``  would act when its top option is a tool with p >= tau; agrees when that is the tool the LLM called.
                 Labels: ``shadow`` and ``guard`` logs, and ``cascade`` steps it did not take. Requests that offered
                 a single tool are left out: there "agreement" only says that the agent calls its one tool, and a
                 cascade fitted on them would move almost every step to the fast tier, whatever the step needs.
                 The global threshold must clear the target on *every* tool at once, so a few unreliable tools hold it
                 up. The same fit also runs per tool (`per_tool` in the output, `tool_tau` in the config snippet): a
                 tool gets its own threshold when its own labelled decisions (those whose top option was that tool)
                 clear the target; the other tools keep the global/default threshold.
``stop``         would accept the fast tier's final reply when p(done) >= tau; agrees when the strong tier also ended
                 the turn. Labels: ``stop: shadow`` logs (every fast-tier final reply is redone by the strong tier).

Routing, reflection and action decisions have no such label in a proxy log (nobody knows what the other tier, or the
agent without the pause, would have done); ``jah fit`` describes them (how often they would fire at the logged tau)
and leaves their thresholds to you.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .points import NO_TOOL, TAU

FITTED = ("tool_select", "stop")


def read_decisions(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "decisions.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run the proxy with decision points in shadow mode first "
                                "(jah serve --preset shadow)")
    out = []
    for ln in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def wilson_lower(k: int, n: int, z: float) -> float:
    """One-sided lower confidence bound of a rate k/n (Wilson score); z = 0 gives k/n."""
    if n == 0:
        return 0.0
    p = k / n
    if z <= 0:
        return p
    z2 = z * z
    centre = p + z2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (centre - spread) / (1 + z2 / n)


def _score(r: dict[str, Any], point: str) -> float | None:
    if point == "tool_select":
        if r.get("choice") in (None, NO_TOOL):
            return None                      # the cascade never acts on "no tool"
        return r.get("confidence")
    p = (r.get("p") or {}).get("done", r.get("p_act"))
    return None if p is None else float(p)


def _agrees(r: dict[str, Any], point: str) -> bool:
    if point == "tool_select":
        return r.get("choice") == r.get("label")
    return r.get("label") == "done"


def n_tools(r: dict[str, Any]) -> int | None:
    """The number of tools a tool-choice question offered (its options without "no tool"), when logged."""
    opts = r.get("options")
    if not isinstance(opts, list):
        return None
    n = int(r.get("n_options") or len(opts))
    return n - (1 if NO_TOOL in opts else 0)


def single_tool(r: dict[str, Any]) -> bool:
    k = n_tools(r)
    return k is not None and k < 2


def labelled(records: Iterable[dict[str, Any]], point: str) -> list[dict[str, Any]]:
    out = []
    for r in records:
        if r.get("point") != point or not r.get("fit_ok") or r.get("error") or r.get("label") is None:
            continue
        if point == "stop" and (r.get("question") != "task_done" or r.get("label") not in ("done", "not_done")):
            continue
        if point == "tool_select" and (r.get("confidence") is None or single_tool(r)):
            continue
        out.append(r)
    return out


def _at(recs: list[dict[str, Any]], point: str, t: float, z: float) -> dict[str, Any]:
    acc = [r for r in recs if (s := _score(r, point)) is not None and s >= t]
    k = sum(1 for r in acc if _agrees(r, point))
    n = len(acc)
    return {"n_accepted": n, "coverage": round(n / len(recs), 4) if recs else 0.0,
            "agreement": round(k / n, 4) if n else None, "lower": round(wilson_lower(k, n, z), 4) if n else None}


def _fit_one_tool(recs: list[dict[str, Any]], point: str, target: float, min_count: int, z: float,
                  denom: int) -> dict[str, Any]:
    """The threshold of one tool, fitted on the labelled decisions whose top option was that tool. ``agreement``
    there is the share of them the LLM actually called, and ``coverage`` is the share of *all* labelled decisions
    the fitted threshold would act on."""
    out: dict[str, Any] = {"n_labelled": len(recs), "tau": None}
    scores = sorted({round(float(r["confidence"]), 4) for r in recs if r.get("confidence") is not None})
    best: dict[str, Any] | None = None
    best_t = None
    for t in scores:
        st = _at(recs, point, t, z)
        st["coverage"] = round(st["n_accepted"] / denom, 4) if denom else 0.0
        if st["lower"] is not None and (best is None or st["lower"] > best["lower"]):
            best, best_t = st, t
        if st["n_accepted"] >= min_count and (st["lower"] or 0.0) >= target:
            nice = min(1.0, math.ceil(t * 100 - 1e-9) / 100)
            st2 = _at(recs, point, nice, z)
            st2["coverage"] = round(st2["n_accepted"] / denom, 4) if denom else 0.0
            if nice != t and st2["n_accepted"] >= min_count and (st2["lower"] or 0.0) >= target:
                st = st2
            out.update({"tau": nice if st is st2 else t, **st})
            return out
    out["note"] = f"no threshold reached {target:.0%} agreement on >= {min_count} decisions"
    if best is not None:
        out["best"] = {"tau": best_t, **best}
    return out


def fit_point(records: list[dict[str, Any]], point: str, target: float = 0.95, min_count: int = 30,
              z: float = 1.645) -> dict[str, Any]:
    recs = labelled(records, point)
    out: dict[str, Any] = {"point": point, "n_records": sum(1 for r in records if r.get("point") == point),
                           "n_labelled": len(recs), "tau": None}
    if point == "tool_select":
        out["single_tool"] = sum(1 for r in records if r.get("point") == point and single_tool(r))
    scores = sorted({round(float(s), 4) for r in recs if (s := _score(r, point)) is not None})
    for t in scores:
        st = _at(recs, point, t, z)
        if st["n_accepted"] >= min_count and (st["lower"] or 0.0) >= target:
            nice = min(1.0, math.ceil(t * 100 - 1e-9) / 100)    # a round threshold, if it still qualifies
            st2 = _at(recs, point, nice, z)
            if nice != t and st2["n_accepted"] >= min_count and (st2["lower"] or 0.0) >= target:
                t, st = nice, st2
            out.update({"tau": t, **st})
            break
    if out["tau"] is None:
        if out.get("single_tool") and len(recs) < min_count:
            out["note"] = (f"{out['single_tool']} decisions offered a single tool (left out: not a choice of tool); "
                           f"{len(recs)} labelled decisions with two or more tools")
        elif len(recs) < min_count:
            out["note"] = f"too few labelled decisions ({len(recs)} < {min_count})"
        else:
            out["note"] = f"no threshold reached {target:.0%} agreement on >= {min_count} decisions"
    if out.get("single_tool") and "single tool" not in out.get("note", ""):
        out["note"] = (out.get("note", "") + "; " if out.get("note") else "") + \
            f"{out['single_tool']} single-tool decisions left out"
    if point == "tool_select":
        # the global threshold must clear the target on every tool at once; per-tool thresholds let each tool
        # earn its own (tools with fewer than min_count labelled decisions are left out: too few to fit)
        by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            if r.get("choice") not in (None, NO_TOOL):
                by_tool[str(r["choice"])].append(r)
        out["per_tool"] = {tool: _fit_one_tool(rs, point, target, min_count, z, len(recs))
                           for tool, rs in sorted(by_tool.items(), key=lambda kv: -len(kv[1]))
                           if len(rs) >= min_count}
    logged = sorted({r.get("tau") for r in recs if r.get("tau") is not None})
    if len(logged) == 1:
        out["at_logged_tau"] = {"tau": logged[0], **_at(recs, point, float(logged[0]), z)}
    return out


def describe(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per point and mode: decision calls, errors, how often the point acted (or would act at the logged tau), the
    latency and the cost."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[f"{r.get('point')}:{r.get('mode')}"].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        ok = [r for r in rs if not r.get("error")]
        lat = sorted(float(r.get("latency_s") or 0) for r in ok)
        fire = [r for r in ok if r.get("p_act") is not None and r.get("tau") is not None
                and float(r["p_act"]) >= float(r["tau"])]
        d = {"calls": len(rs), "errors": len(rs) - len(ok), "acted": sum(1 for r in rs if r.get("acted")),
             "would_fire_at_logged_tau": round(len(fire) / len(ok), 4) if ok and rs[0].get("act_option") else None,
             "latency_p50_s": lat[len(lat) // 2] if lat else None,
             "latency_p99_s": lat[min(len(lat) - 1, round(0.99 * (len(lat) - 1)))] if lat else None,
             "latency_max_s": lat[-1] if lat else None,
             "cost_usd": round(sum(float(r.get("cost_usd") or 0) for r in rs), 6)}
        rechecks = [r for r in rs if r.get("recheck")]
        if rechecks:
            d["rechecks_changed"] = round(sum(1 for r in rechecks if r["recheck"] != "same") / len(rechecks), 4)
            d["rechecks_still_risky"] = round(sum(1 for r in rechecks if r.get("recheck_risky")) / len(rechecks), 4)
        out[g] = d
    return out


def fit(records: list[dict[str, Any]], target: float = 0.95, min_count: int = 30, z: float = 1.645,
        model: str | None = None) -> dict[str, Any]:
    names = sorted({str(r.get("model_name")) for r in records})
    if model is not None:
        records = [r for r in records if r.get("model_name") == model]
    return {"target": target, "min_count": min_count, "z": z, "model_names": names, "model": model,
            "n_records": len(records), "points": {p: fit_point(records, p, target, min_count, z) for p in FITTED},
            "describe": describe(records)}


def _pct(x: Any) -> str:
    return "-" if x is None else f"{100 * float(x):.1f}%"


def table(res: dict[str, Any]) -> str:
    rows = ["| point | labelled | fitted tau | acts on | agreement | lower bound | note |",
            "|---|---|---|---|---|---|---|"]
    for p, f in res["points"].items():
        tau = "-" if f["tau"] is None else f"{f['tau']:g}"
        rows.append(f"| {p} | {f['n_labelled']} | {tau} | {_pct(f.get('coverage'))} | {_pct(f.get('agreement'))} | "
                    f"{_pct(f.get('lower'))} | {f.get('note', '')} |")
    pt = res["points"].get("tool_select", {}).get("per_tool")
    if pt:
        rows += ["", "| per-tool (tool_select) | labelled | fitted tau | acts on | agreement | lower bound | note |",
                 "|---|---|---|---|---|---|---|"]
        for tool, f in pt.items():
            tau = "-" if f["tau"] is None else f"{f['tau']:g}"
            note = f.get("note", "")
            if f["tau"] is None and f.get("best"):
                b = f["best"]
                note += (f" (best: {_pct(b.get('agreement'))} agreement, lower bound {_pct(b.get('lower'))}, "
                         f"at tau {b.get('tau'):g} on {b.get('n_accepted')} decisions)")
            rows.append(f"| {tool} | {f['n_labelled']} | {tau} | {_pct(f.get('coverage'))} | "
                        f"{_pct(f.get('agreement'))} | {_pct(f.get('lower'))} | {note} |")
    rows += ["", "| point:mode | calls | errors | acted | fires at logged tau | p50 latency s | cost $ |",
             "|---|---|---|---|---|---|---|"]
    for g, d in res["describe"].items():
        rows.append(f"| {g} | {d['calls']} | {d['errors']} | {d['acted']} | {_pct(d['would_fire_at_logged_tau'])} | "
                    f"{'-' if d['latency_p50_s'] is None else round(d['latency_p50_s'], 3)} | {d['cost_usd']} |")
    return "\n".join(rows)


def _ykey(k: str) -> str:
    """A YAML bare key if the tool name allows it, else single-quoted."""
    if k and all(ch.isalnum() or ch in "_.-" for ch in k):
        return k
    return "'" + k.replace("'", "''") + "'"


def snippet(res: dict[str, Any], source: str = "", era_note: str = "") -> str:
    head = (f"# jah fit {source}: target {res['target']:.0%} agreement with the LLM on >= {res['min_count']} "
            f"decisions, z = {res['z']:g} ({date.today().isoformat()})")
    if era_note:
        head += f"\n# era filter: {era_note}"
    if res.get("model"):
        head += f"\n# model name: {res['model']} (put this under models: {{{res['model']}: {{decisions: ...}}}})"
    elif len(res.get("model_names") or []) > 1:
        head += f"\n# pooled over model names {', '.join(res['model_names'])} (use --model to fit one)"
    lines = [head, "decisions:"]
    for p, f in res["points"].items():
        # only per-tool thresholds that differ from the effective global one: the rest change nothing
        base = f["tau"] if f["tau"] is not None else TAU[p]
        fitted = {t: v["tau"] for t, v in (f.get("per_tool") or {}).items()
                  if v.get("tau") is not None and v["tau"] != base}
        if f["tau"] is not None:
            tau, note = f["tau"], (f"{f['n_labelled']} labelled: acts on {_pct(f['coverage'])}, "
                                   f"{_pct(f['agreement'])} agree (lower bound {_pct(f['lower'])})")
        elif fitted:
            # no single global threshold qualified: the per-tool thresholds act, the rest stay at the default
            tau, note = base, (f"{f.get('note')}; per-tool thresholds only: the other tools keep the LLM "
                               f"deciding (tau {base:g}, almost never)")
        else:
            keep = "mode: strong" if p == "stop" else "mode: off or shadow"
            lines.append(f"  # {p}: {f.get('note')}: keep the LLM deciding ({keep})")
            continue
        line = f"  {p}: {{tau: {tau:g}"
        if fitted:
            line += ", tool_tau: {" + ", ".join(f"{_ykey(t)}: {v:g}" for t, v in sorted(fitted.items())) + "}"
        lines.append(line + "}    # " + note)
    return "\n".join(lines) + "\n"
