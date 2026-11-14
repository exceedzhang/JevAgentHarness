"""Drex arm of the A/B: score Drex v1.5 (nace-ai/drex-v1.5) on the SAME routing decisions.

Uses the Drex repo's own ``serve.py`` (``POST /v1/systemone``), the reference server for Drex v1.5, so the
numbers describe what the Drex runtime does. Each decision is one single System One request (no batch packing)
so the measured latency is the per-step cost, matching the JAH arm.

Prerequisites (done by run_all.sh): the Drex repo is cloned at ``--drex-repo`` with its venv, and the v1.5
weights are downloaded at ``--weights``.

    python research/ab_test/run_drex.py \
        --drex-repo /path/to/drex-decision-models \
        --weights /path/to/drex-v1.5 \
        --data research/data/routing_gold_v2.jsonl \
        --out research/ab_test/out/drex.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ab_common import QUESTION_ID, dump, load_decisions, score, systemone_request  # noqa: E402


def _http_json(url: str, payload: dict | None = None, timeout: float = 600.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _wait_health(base: str, timeout_s: int = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if _http_json(f"{base}/health")["status"] == "ok":
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"Drex /health did not come up at {base} within {timeout_s}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B arm: Drex v1.5 on routing decisions")
    ap.add_argument("--drex-repo", required=True, help="cloned drex-decision-models repo (has serve.py + venv)")
    ap.add_argument("--weights", required=True, help="downloaded nace-ai/drex-v1.5 dir (safetensors + head.pt)")
    ap.add_argument("--python", default=None, help="python to run serve.py (default: <drex-repo>/.venv/bin/python)")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "out" / "drex.json"))
    a = ap.parse_args()

    py = a.python or str(Path(a.drex_repo) / ".venv" / "bin" / "python")
    base = f"http://127.0.0.1:{a.port}"

    records = load_decisions(a.data)
    if a.limit:
        records = records[: a.limit]

    server = subprocess.Popen(
        [py, "serve.py", "--model", a.weights, "--port", str(a.port)],
        cwd=a.drex_repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    results: dict[str, dict] = {}
    try:
        _wait_health(base)
        t_start = time.perf_counter()
        for i, r in enumerate(records):
            req = systemone_request(r)
            t0 = time.perf_counter()
            resp = _http_json(f"{base}/v1/systemone", req)
            dt = (time.perf_counter() - t0) * 1000.0
            if i < a.warmup:
                continue
            probs = resp["answers"][QUESTION_ID]["probabilities"]
            results[r["id"]] = {
                "probs": {k: float(v) for k, v in probs.items()},
                "latency_ms": dt,
                "drex_latency_ms": resp.get("latency_ms"),
            }
        wall = (time.perf_counter() - t_start) * 1000.0
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

    metrics = score(records, results)
    metrics["wall_ms_all"] = round(wall, 1)
    metrics["model"] = "nace-ai/drex-v1.5"
    dump({"metrics": metrics, "per": results}, a.out)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
