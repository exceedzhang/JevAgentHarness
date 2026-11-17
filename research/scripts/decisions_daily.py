#!/usr/bin/env python3
"""Daily pull of decisions.jsonl from the local JAH container + dead-pool accumulation tracking.

Deterministic (no LLM): pull -> count (total + per-tool, esp. the dead pools) -> delta vs last
state -> append to a dated log -> flag "retrain candidate" when a dead pool's CUMULATIVE real-sample
count crosses its threshold.

Why cumulative total (not "since a date"): a retrain would mine boundary pairs from ALL real
<tool> decision-states in the log, so the pool size that matters is the running total. The daily
delta shows how fast each dead pool is accumulating in production.

The dead pools (glob/webfetch/grep/compress/not_done) are exhausted in the *training* set
(train_v4 had glob=3, webfetch=2); this tracks the REAL production samples accumulating so we know
when there's enough to justify a retrain round on a given tool.
"""
import json, subprocess, datetime, os

DOCKER = "/usr/local/bin/docker"
CONTAINER = "litellm-jah-litellm-jah-1"
SRC = "/root/.jah/litellm-logs/decisions.jsonl"
HOME = os.path.expanduser("~")
STATE = os.path.join(HOME, ".jah-pull-state.json")
LOG = os.path.join(HOME, "jah-decisions-daily", "log.jsonl")
DEAD_POOLS = ["glob", "webfetch", "grep", "compress", "not_done"]
# cumulative real-sample thresholds that flag "enough to consider a retrain round on this tool"
THRESH = {"glob": 30, "webfetch": 30, "grep": 50, "compress": 80, "not_done": 30}

CANON = {"bash": "shell", "shell_run": "shell", "shell_wait": "shell",
         "ask_human": "ask_user_question", "question": "ask_user_question",
         "web_fetch": "webfetch", "web_search": "websearch"}


def pull():
    p = subprocess.run([DOCKER, "exec", CONTAINER, "sh", "-c", f"cat {SRC}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("docker pull failed: " + p.stderr[:300])
    return [l for l in p.stdout.splitlines() if l.strip()]


def day_of(ts_raw):
    if isinstance(ts_raw, (int, float)):
        return datetime.datetime.fromtimestamp(ts_raw, datetime.timezone.utc).strftime("%Y-%m-%d")
    if isinstance(ts_raw, str):
        return ts_raw[:10]
    return ""


def main():
    lines = pull()
    now = datetime.datetime.now()
    total = len(lines)
    all_tool, today_tool, per_day = {}, {}, {}
    n_routing = n_select = n_stop = 0
    today = now.strftime("%Y-%m-%d")
    for l in lines:
        try:
            o = json.loads(l)
        except Exception:
            continue
        pt = o.get("point")
        if pt == "routing":
            n_routing += 1
        elif pt == "stop":
            n_stop += 1
        if pt != "tool_select":
            continue
        n_select += 1
        lab = o.get("label")
        if not lab:
            continue
        t = CANON.get(lab, lab)
        all_tool[t] = all_tool.get(t, 0) + 1
        d = day_of(o.get("ts"))
        per_day[d] = per_day.get(d, 0) + 1
        if d == today:
            today_tool[t] = today_tool.get(t, 0) + 1

    prev = {}
    if os.path.exists(STATE):
        try:
            prev = json.load(open(STATE))
        except Exception:
            prev = {}
    prev_total = prev.get("total", 0)
    prev_tool = prev.get("per_tool", {})
    delta_total = total - prev_total
    delta_tool = {t: all_tool.get(t, 0) - prev_tool.get(t, 0) for t in set(all_tool) | set(prev_tool)}

    json.dump({"ts": now.isoformat(timespec="seconds"), "total": total, "per_tool": all_tool}, open(STATE, "w"))
    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    flags = [f"RETRAIN CANDIDATE: {t} (cumulative real samples {all_tool.get(t,0)} >= threshold {THRESH[t]})"
             for t in THRESH if all_tool.get(t, 0) >= THRESH[t]]
    entry = {"ts": now.isoformat(timespec="seconds"), "total": total, "delta_total": delta_total,
             "points": {"routing": n_routing, "tool_select": n_select, "stop": n_stop},
             "delta_tool": delta_tool, "today_tool": today_tool,
             "cum_dead_pools": {t: all_tool.get(t, 0) for t in DEAD_POOLS},
             "last5_days_total": {d: per_day.get(d, 0) for d in sorted(per_day)[-5:]},
             "flags": flags}
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[{now.isoformat(timespec='seconds')}] total={total} (delta vs last pull {delta_total:+d})")
    print(f"  points: routing {n_routing} / tool_select {n_select} / stop {n_stop}")
    print(f"  last 5 days total rows: " + ", ".join(f"{d}={per_day.get(d,0)}" for d in sorted(per_day)[-5:]))
    print("dead-pool cumulative real samples (the pool a retrain would mine):")
    for t in DEAD_POOLS:
        print(f"  {t:12s} total={all_tool.get(t,0):5d}  delta={delta_tool.get(t,0):+d}  today={today_tool.get(t,0):d}  threshold={THRESH.get(t,'-')}")
    for fl in flags:
        print("FLAG: " + fl)
    if not flags:
        print("(no retrain-candidate threshold reached yet)")


if __name__ == "__main__":
    main()
