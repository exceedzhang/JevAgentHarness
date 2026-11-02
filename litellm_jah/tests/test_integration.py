"""Integration test: a real litellm completion against a fake OpenAI-compatible server, with the JAH
step logger registered through litellm's real success-callback machinery. This proves the logger sees
the genuine event payload (top-level messages/model/usage, no "data" wrapper) end to end.

The pre/post-call hooks are proxy-only machinery (the proxy's guardrail runner invokes them; a bare
Router/acompletion does not); their payload contract was validated against the real proxy in the
docker e2e, and their logic is unit-tested with the documented shapes.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import litellm
from litellm_jah.backend import JevBackend
from litellm_jah.config import JAHCfg
from litellm_jah.engine import JAHRouter
from litellm_jah.logger import JahStepLogger
from litellm_jah.postcall import JahPostCall
from litellm_jah.precall import JahPreCall

REPLY = "ok, done."
IN_T, OUT_T = 11, 4


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        out = {
            "id": "cmpl-test", "object": "chat.completion", "created": 0, "model": "fake-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": REPLY},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": IN_T, "completion_tokens": OUT_T,
                      "total_tokens": IN_T + OUT_T},
        }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_real_completion_fires_the_step_logger(tmp_path, run):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"

    async def main():
        cfg = JAHCfg(decision={"type": "rules"}, verify_finish=False, log_dir=str(tmp_path))
        backend = JevBackend(base_url=base, model="fake-strong", key_env=None, timeout_s=5.0)
        engine = JAHRouter(cfg, backend=backend)
        logger = JahStepLogger(engine=engine)
        litellm._async_success_callback.append(logger)
        try:
            # pre-call first (as the proxy runs it): the decision row and the routed-step record
            m1 = [{"role": "user", "content": "say ok"}]
            md1 = {"session_id": "s-logger"}
            await JahPreCall(engine=engine).async_pre_call_hook(None, None, {
                "model": "jah-auto", "messages": m1, "metadata": md1,
                "tools": [{"type": "function", "function": {"name": "bash", "description": "d",
                                                        "parameters": {}}}]} , "completion")
            # the real entry point, the real success event; session id only, as a bare client sends
            resp = await litellm.acompletion(
                model="openai/fake-model", api_base=base, api_key="sk-test",
                messages=m1, metadata=md1,
            )
            assert resp.choices[0].message.content == REPLY
            # the success event is fire-and-forget: poll for the row the logger writes in the
            # background task
            rows = []
            for _ in range(100):
                await asyncio.sleep(0.05)
                path = tmp_path / "steps.jsonl"
                if path.exists():
                    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
                if rows:
                    break
            assert len(rows) == 1, rows
            assert rows[0]["model_name"] == "jah-auto"  # the group the plugin serves
            assert rows[0]["step"] == 0
            assert rows[0]["input_tokens"] == IN_T and rows[0]["output_tokens"] == OUT_T
            assert rows[0]["calls"][0]["model"] == "fake-model"  # the serving model from the real event
        finally:
            if logger in litellm._async_success_callback:
                litellm._async_success_callback.remove(logger)
            await engine.close()

        # the pre/post-call pair driven with the shapes the proxy passes (client group in pre-call,
        # deployment model + model_group stamp in post-call), with the real response object
        cfg2 = JAHCfg(decision={"type": "rules"}, verify_finish=False, log_dir=str(tmp_path))
        engine2 = JAHRouter(cfg2, backend=JevBackend(base_url=base, model="fake-strong",
                                                     key_env=None, timeout_s=5.0))
        pre, post = JahPreCall(engine=engine2), JahPostCall(engine=engine2)
        m = [{"role": "user", "content": "say ok"}]
        md = {"session_id": "s-hooks"}
        await pre.async_pre_call_hook(None, None, {
            "model": "jah-auto", "messages": m, "metadata": md,
            "tools": [{"type": "function", "function": {"name": "bash", "description": "d",
                                                        "parameters": {}}}]}, "completion")
        resp2 = await litellm.acompletion(
            model="openai/fake-model", api_base=base, api_key="sk-test",
            messages=m + [{"role": "assistant", "content": REPLY}],
            metadata=md,
        )
        await post.async_post_call_success_deployment_hook(
            {"model": "openai/fake-model", "messages": m,
             "metadata": {**md, "model_group": "jah-auto"},
             "tools": [{"type": "function", "function": {"name": "bash", "description": "d",
                                                         "parameters": {}}}]} ,
            resp2, "completion")
        sess = await engine2.store.get("s-hooks")  # the explicit session id is the key
        assert set(sess.step_points) == {0}  # the post-call settled the step
        rec = sess.decided.get(0)
        assert rec is not None and rec.get("tier") == "strong"  # the routed record post-call consumed
        await engine2.close()

    try:
        run(asyncio.wait_for(main(), timeout=30))
    finally:
        server.shutdown()
