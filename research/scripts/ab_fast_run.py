"""A/B: run the FAST model (Qwen3.8-27B) on each routing state, predict the next tool,
and record it for scoring against the agent's actual tool. Runs on the GPU box (the 27B
is served at http://localhost:8000/v1). The state is the classifier's text projection
(state + tool names only, no full tool schemas), so this is a mild handicap vs.
production; the point is the ranking (does p_hard predict failure), not the absolute rate.

Usage: AB_KEY=<tunnel key> python ab_fast_run.py <in.jsonl> <out.jsonl>
Env: AB_CONC (default 4), AB_LIMIT (default 0 = all).
"""
import json, os, sys, time, urllib.request, concurrent.futures as cf

HOST = os.environ.get("AB_HOST", "http://localhost:8000/v1")
KEY = os.environ["AB_KEY"]
MODEL = os.environ.get("AB_MODEL", "Qwen/Qwen3.8-27B")
CONC = int(os.environ.get("AB_CONC", "4"))
LIMIT = int(os.environ.get("AB_LIMIT", "0"))
MAXTOK = 1024
# Qwen3.8-27B is a reasoning model: with thinking on it spends ~600 tokens reasoning before
# the tool call (and truncates at a small budget). Disable thinking for this prediction task.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def tools_payload(tools):
    out = []
    for t in tools:
        desc = ("Reply to the user with text. Use when the correct next action is to answer, "
                "not to call a tool.") if t == "reply_without_tool" else "Tool: " + t
        out.append({"type": "function",
                    "function": {"name": t, "description": desc,
                                 "parameters": {"type": "object", "properties": {}}}})
    return out


def call(row):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": ("You are an AI agent with access to tools. From the task and the steps "
                         "already taken, choose the single tool to call next. If the correct next "
                         "action is to reply to the user with text, use reply_without_tool.")},
            {"role": "user", "content": row["state"] + "\n\nChoose the single tool to call next."},
        ],
        "tools": tools_payload(row["tools"]),
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": MAXTOK,
    }
    body.update(NO_THINK)
    data = json.dumps(body).encode()
    req = urllib.request.Request(HOST + "/chat/completions", data=data,
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
            m = (d.get("choices") or [{}])[0].get("message") or {}
            tcs = m.get("tool_calls") or []
            pred = tcs[0]["function"]["name"] if tcs else "TEXT:" + (m.get("content") or "")[:60]
            return {"id": row["id"], "pred": pred, "ok": True,
                    "content": (m.get("content") or "")[:200]}
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"id": row["id"], "pred": None, "ok": False, "err": str(e)[:140]}
            time.sleep(2 * (attempt + 1))


def main():
    IN, OUT = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for l in open(IN) if l.strip()]
    if LIMIT:
        rows = rows[:LIMIT]
    results = []
    with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = {ex.submit(call, r): r for r in rows}
        done = 0
        for f in cf.as_completed(futs):
            res = f.result()
            results.append(res)
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(rows)}", flush=True)
    order = {r["id"]: i for i, r in enumerate(rows)}
    results.sort(key=lambda x: order.get(x["id"], 10 ** 9))
    with open(OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("done", len(results), "ok", sum(1 for r in results if r["ok"]))


if __name__ == "__main__":
    main()
