"""Local mock gateway hiding models behind anonymous IDs. Serves both
POST /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic), streaming.

Used for rehearsal with the real client, so the ledger, retry rules and stream
parsing are exercised exactly as on contest day.

    python3 -m sim.gateway --world synthetic --port 8765 --seed 3
    python3 -m sim.gateway --world replay --split test --port 8765

Then:  CONTEST_API_KEY=x python3 run_contest.py --base-url http://127.0.0.1:8765/v1
The answer key is written to runs/gateway_truth.json, not printed.
"""
import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from candidates import CANDIDATES
from config import RUNS
from probes.papers import MAIN, TONE, load_papers
from reference.configs import configs


class Gateway:
    def __init__(self, world, papers, assign, *, seed=0, strict=True, time_scale=0.0, fail=None):
        self.world, self.assign = world, assign
        self.by_prompt = {p.prompt: p for p in papers.values()}
        self.rng = random.Random(seed)
        self.strict, self.time_scale = strict, time_scale
        self.fail = dict(fail or {})    # model id -> HTTP status to return once
        self.count = 0
        self.lock = threading.Lock()

    def handle(self, body):
        """Returns (http_status, record or error dict). Every call counts, like the contest."""
        with self.lock:
            self.count += 1
            mid = body.get("model")
            if mid in self.fail:
                return self.fail.pop(mid), {"error": {"message": "injected failure"}}
            if mid not in self.assign:
                return 404, {"error": {"message": f"model {mid} not found"}}
            if self.strict:
                # mirrors documented upstream behaviour: these get a 400 on some backends
                if body.get("n", 1) != 1:
                    return 400, {"error": {"message": "n must be 1"}}
                for bad in ("temperature", "top_p", "response_format", "logprobs"):
                    if bad in body:
                        return 400, {"error": {"message": f"unsupported parameter: {bad}"}}
            user = next((m["content"] for m in body.get("messages", []) if m.get("role") == "user"), "")
            if isinstance(user, list):   # Anthropic content blocks
                user = "".join(b.get("text", "") for b in user if isinstance(b, dict))
            paper = self.by_prompt.get(user)
            if paper is None:
                return 400, {"error": {"message": "mock gateway only answers frozen probe papers"}}
            key, cfg = self.assign[mid]
            return 200, self.world.sample(key, paper, cfg, self.rng)


def make_handler(gw):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": {"message": "bad json"}})
            if self.path.endswith("/messages"):
                return self._messages(body)
            if not self.path.endswith("/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            status, rec = gw.handle(body)
            if status != 200:
                return self._json(status, rec)
            if rec.get("status") != "ok":
                return self._json(rec.get("http_status") or 503, {"error": {"message": rec.get("error")}})
            rid, mid, text = "chatcmpl-mock%06d" % gw.count, body["model"], rec["text"]
            if not body.get("stream"):
                return self._json(200, {"id": rid, "object": "chat.completion", "model": mid,
                                        "choices": [{"index": 0, "finish_reason": rec["finish_reason"],
                                                     "message": {"role": "assistant", "content": text}}],
                                        "usage": rec["usage"]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(obj):
                self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()

            base = {"id": rid, "object": "chat.completion.chunk", "model": mid}
            time.sleep((rec.get("ttft") or 0) * gw.time_scale)
            step = 48
            pause = max((rec.get("duration") or 0) - (rec.get("ttft") or 0), 0) * gw.time_scale / max(len(text) / step, 1)
            for i in range(0, len(text), step):
                send({**base, "choices": [{"index": 0, "delta": {"content": text[i:i + step]}, "finish_reason": None}]})
                if pause:
                    time.sleep(pause)
            send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": rec["finish_reason"]}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                send({**base, "choices": [], "usage": rec["usage"]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _messages(self, body):
            """Anthropic Messages streaming, as /v1/messages sends it."""
            if "max_tokens" not in body:
                gw.count += 1
                return self._json(400, {"type": "error", "error": {"type": "invalid_request_error",
                                                                   "message": "max_tokens: Field required"}})
            status, rec = gw.handle(body)
            if status != 200:
                return self._json(status, {"type": "error", "error": {"message": str(rec.get("error"))}})
            if rec.get("status") != "ok":
                return self._json(rec.get("http_status") or 503,
                                  {"type": "error", "error": {"message": str(rec.get("error"))}})
            mid, text = body["model"], rec.get("text") or ""
            usage = rec.get("usage") or {}
            stop = {"stop": "end_turn", "length": "max_tokens"}.get(rec.get("finish_reason"),
                                                                     rec.get("finish_reason") or "end_turn")
            thinking = "thinking" in (rec.get("block_types") or [])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(event, obj):
                self.wfile.write(f"event: {event}\ndata: ".encode()
                                 + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()

            send("message_start", {"type": "message_start", "message": {
                "id": "msg_mock%06d" % gw.count, "type": "message", "role": "assistant", "model": mid,
                "content": [], "stop_reason": None,
                "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": 1}}})
            time.sleep((rec.get("ttft") or 0) * gw.time_scale)
            idx = 0
            if thinking:
                send("content_block_start", {"type": "content_block_start", "index": idx,
                                             "content_block": {"type": "thinking", "thinking": ""}})
                send("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {
                    "type": "thinking_delta", "thinking": "." * max(1, rec.get("reasoning_chars") or 1)}})
                if rec.get("thinking_signature"):
                    send("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                 "delta": {"type": "signature_delta", "signature": "mock"}})
                send("content_block_stop", {"type": "content_block_stop", "index": idx})
                idx += 1
            send("content_block_start", {"type": "content_block_start", "index": idx,
                                         "content_block": {"type": "text", "text": ""}})
            for i in range(0, len(text), 48):
                send("content_block_delta", {"type": "content_block_delta", "index": idx,
                                             "delta": {"type": "text_delta", "text": text[i:i + 48]}})
            send("content_block_stop", {"type": "content_block_stop", "index": idx})
            send("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop},
                                   "usage": {"output_tokens": usage.get("completion_tokens", 0)}})
            send("message_stop", {"type": "message_stop"})

    return H


def serve(gw, port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(gw))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="synthetic", choices=["synthetic", "replay"])
    ap.add_argument("--split", default="test", help="dev, test or a red-team split such as rt5")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="claude-opus-5")
    ap.add_argument("--time-scale", type=float, default=0.0, help="1.0 replays recorded timing")
    ap.add_argument("--routes", default="ppapi_msg", help="replay answers collected on these routes")
    a = ap.parse_args()

    papers = load_papers()
    if a.world == "synthetic":
        from sim.synthetic import SyntheticWorld
        world = SyntheticWorld(seed=a.seed)
        pool = world.keys()
    else:
        from reference.store import load_records
        from sim.evaluate import ReplayWorld
        world = ReplayWorld(load_records(a.routes.split(",")), papers)
        pool = [k for k in world.keys() if world.has(k, MAIN, a.split) and world.has(k, TONE, a.split)]
    rng = random.Random(a.seed)
    truth = [a.target] + rng.sample([k for k in pool if k != a.target], 9)
    rng.shuffle(truth)
    ids = [f"model-{c}" for c in "abcdefghij"]
    assign = {i: (t, rng.choice(configs(a.split))) for i, t in zip(ids, truth)}
    RUNS.mkdir(exist_ok=True)
    (RUNS / "gateway_truth.json").write_text(json.dumps(
        {i: {"truth": t, "name": CANDIDATES[t]["name"], "version": CANDIDATES[t]["version"],
             "label": CANDIDATES[t].get("label"), "config": c["id"]}
         for i, (t, c) in assign.items()}, indent=1))
    gw = Gateway(world, papers, assign, seed=a.seed, time_scale=a.time_scale)
    serve(gw, a.port)
    print(f"mock gateway on http://127.0.0.1:{a.port}/v1  ids: {ids[0]} .. {ids[-1]}")
    print("answer key: runs/gateway_truth.json (do not open before the run)")
    try:
        while True:
            time.sleep(5)
            print(f"requests counted so far: {gw.count}", end="\r")
    except KeyboardInterrupt:
        print(f"\nrequests counted: {gw.count}")


if __name__ == "__main__":
    main()
