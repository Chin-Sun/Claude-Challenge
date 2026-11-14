"""OpenAI-compatible chat client built on http.client.

Retry rule: retry only while the TCP/TLS connection has not been established
(refused, DNS, handshake). Once the request is sent, never retry: any HTTP
response, a read timeout or a broken stream may already have been counted by
the server.
"""
import http.client
import json
import os
import re
import socket
import ssl
import time
import urllib.parse

from config import REQUEST_PROFILES


def _new_conn(base_url, connect_timeout):
    u = urllib.parse.urlparse(base_url)
    port = u.port or (443 if u.scheme == "https" else 80)
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(u.hostname, port, timeout=connect_timeout,
                                           context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(u.hostname, port, timeout=connect_timeout)
    return conn, u.path.rstrip("/") + "/chat/completions"


def _absorb_anthropic(rec, j, t_rel):
    """Fold one Anthropic Messages event (stream event or full body) into the record.
    Usage is normalised to prompt_tokens / completion_tokens so features stay comparable."""
    if not isinstance(j, dict):
        return
    kind = j.get("type")

    def usage(u):
        if not isinstance(u, dict):
            return
        cur = rec["usage"]
        for k, v in u.items():
            if v is not None:
                cur[k] = v
        if "input_tokens" in cur:
            cur["prompt_tokens"] = (cur.get("input_tokens") or 0) + (cur.get("cache_read_input_tokens") or 0) \
                + (cur.get("cache_creation_input_tokens") or 0)
        if "output_tokens" in cur:
            cur["completion_tokens"] = cur.get("output_tokens") or 0

    def block(b):
        if not isinstance(b, dict):
            return
        rec["block_types"] = sorted(set(rec.get("block_types", [])) | {str(b.get("type"))})
        text = b.get("text")
        think = b.get("thinking")
        if isinstance(think, str) and think:
            rec["reasoning_chars"] += len(think)
            if rec["ttft"] is None:
                rec["ttft"] = t_rel
        if b.get("signature") or b.get("type") == "signature_delta":
            rec["thinking_signature"] = True
        if isinstance(text, str) and text:
            rec["text"] += text
            if rec["ttft"] is None:
                rec["ttft"] = t_rel
            if rec["ttft_content"] is None:
                rec["ttft_content"] = t_rel

    if kind == "error" or (kind is None and j.get("error")):
        rec["error"] = json.dumps(j.get("error"), ensure_ascii=False)[:2000]
        return
    msg = j.get("message") if kind == "message_start" else (j if kind == "message" else None)
    if msg:
        rec["resp_id"] = rec["resp_id"] or msg.get("id")
        rec["resp_model"] = rec["resp_model"] or msg.get("model")
        usage(msg.get("usage"))
        for b in msg.get("content") or []:
            block(b)
        if msg.get("stop_reason"):
            rec["finish_reason"] = msg["stop_reason"]
    elif kind == "content_block_start":
        block(j.get("content_block"))
    elif kind == "content_block_delta":
        block(j.get("delta"))
    elif kind == "message_delta":
        usage(j.get("usage"))
        if (j.get("delta") or {}).get("stop_reason"):
            rec["finish_reason"] = j["delta"]["stop_reason"]


def _blank(model, tag):
    return {"model_id": model, "tag": tag, "status": "not_sent", "http_status": None,
            "text": "", "reasoning_chars": 0, "usage": {}, "ttft": None, "ttft_content": None,
            "duration": None, "finish_reason": None, "resp_id": None, "resp_model": None,
            "system_fingerprint": None, "headers": {}, "error": None, "counted": False,
            "sent_at": None}


def _absorb(rec, j, t_rel):
    """Fold one response object (stream chunk or full body) into the record."""
    if not isinstance(j, dict):
        return
    if j.get("id") and not rec["resp_id"]:
        rec["resp_id"] = j["id"]
    if j.get("model") and not rec["resp_model"]:
        rec["resp_model"] = j["model"]
    if j.get("system_fingerprint"):
        rec["system_fingerprint"] = j["system_fingerprint"]
    if isinstance(j.get("usage"), dict):
        rec["usage"] = j["usage"]
    if j.get("error") and not rec["error"]:
        rec["error"] = json.dumps(j["error"], ensure_ascii=False)[:2000]
    for ch in j.get("choices") or []:
        part = ch.get("delta") or ch.get("message") or {}
        content = part.get("content")
        reasoning = part.get("reasoning_content") or part.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            rec["reasoning_chars"] += len(reasoning)
            if rec["ttft"] is None:
                rec["ttft"] = t_rel
        if isinstance(content, str) and content:
            rec["text"] += content
            if rec["ttft"] is None:
                rec["ttft"] = t_rel
            if rec["ttft_content"] is None:
                rec["ttft_content"] = t_rel
        if ch.get("finish_reason"):
            rec["finish_reason"] = ch["finish_reason"]


def chat(base_url, api_key, model, messages, *, profile=None, ledger=None, tag="",
         raw_path=None, connect_timeout=30, read_timeout=600, connect_retries=3,
         protocol="openai"):
    """protocol: "openai" -> POST <base>/chat/completions; "anthropic" -> POST <base>/messages."""
    anthropic = protocol == "anthropic"
    if profile is None:
        profile = REQUEST_PROFILES[protocol]
    rec = _blank(model, tag)
    rec["protocol"] = protocol
    absorb = _absorb_anthropic if anthropic else _absorb
    if anthropic:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        payload = {"model": model, "messages": [m for m in messages if m["role"] != "system"], **profile}
        if system:
            payload["system"] = system
        headers = {"Content-Type": "application/json", "x-api-key": api_key,
                   "anthropic-version": "2023-06-01"}
    else:
        payload = {"model": model, "messages": messages, **profile}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    headers["Accept"] = "text/event-stream" if profile.get("stream") else "application/json"
    body = json.dumps(payload, ensure_ascii=False).encode()

    if ledger is not None and ledger.remaining <= 0:
        rec["error"] = "budget exhausted"
        return rec

    conn = None
    for attempt in range(connect_retries + 1):
        conn, path = _new_conn(base_url, connect_timeout)
        if anthropic:
            path = path.rsplit("/chat/completions", 1)[0] + "/messages"
        try:
            conn.connect()
            break
        except (OSError, ssl.SSLError, socket.gaierror) as e:
            # Nothing was sent: safe to retry, not counted.
            rec["error"] = f"connect: {type(e).__name__}: {e}"
            conn.close()
            conn = None
            if attempt < connect_retries:
                time.sleep(min(2 ** attempt, 8))
    if conn is None:
        return rec

    seq = ledger.reserve(model, tag) if ledger is not None else None
    rec["counted"] = True
    rec["error"] = None
    raw = open(raw_path, "wb") if raw_path else None
    t0 = time.monotonic()
    rec["sent_at"] = time.time()
    try:
        conn.sock.settimeout(read_timeout)
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        rec["http_status"] = resp.status
        rec["headers"] = {k.lower(): v for k, v in resp.getheaders()}
        ctype = rec["headers"].get("content-type", "")
        if resp.status != 200 or "event-stream" not in ctype:
            data = resp.read()
            if raw:
                raw.write(data)
            try:
                absorb(rec, json.loads(data.decode("utf-8", "replace")), time.monotonic() - t0)
            except ValueError:
                rec["error"] = data.decode("utf-8", "replace")[:2000]
            rec["status"] = "ok" if resp.status == 200 else "http_error"
        else:
            while True:
                line = resp.readline()
                if not line:
                    break
                if raw:
                    raw.write(line)
                    raw.flush()
                s = line.decode("utf-8", "replace").strip()
                if not s.startswith("data:"):
                    continue
                s = s[5:].strip()
                if s == "[DONE]":
                    break
                try:
                    absorb(rec, json.loads(s), time.monotonic() - t0)
                except ValueError:
                    continue
            rec["status"] = "ok"
        if rec["status"] == "ok" and rec["error"] and not rec["text"] and not rec["finish_reason"]:
            rec["status"] = "upstream_error"   # e.g. "servers overloaded" sent as a stream event
    except socket.timeout as e:
        rec["status"], rec["error"] = "timeout", f"{type(e).__name__}: {e}"
    except (OSError, http.client.HTTPException) as e:
        rec["status"], rec["error"] = "stream_error", f"{type(e).__name__}: {e}"
    finally:
        rec["duration"] = time.monotonic() - t0
        if raw:
            raw.close()
        conn.close()
        if ledger is not None:
            ledger.outcome(seq, rec["status"], rec["http_status"])
    return rec


class HttpTransport:
    """ask(model_id, paper) -> record. Saves every record and raw stream to disk."""

    def __init__(self, base_url, api_key, *, ledger=None, out_dir=None, system=None,
                 profile=None, read_timeout=600, protocol="openai"):
        self.base_url, self.api_key, self.protocol = base_url, api_key, protocol
        self.ledger, self.out_dir, self.system = ledger, out_dir, system
        self.profile, self.read_timeout = profile, read_timeout

    def ask(self, model_id, paper, name=None):
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": paper.prompt})
        raw_path = None
        stem = name or f"{model_id}__{paper.name}"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            raw_path = os.path.join(self.out_dir, stem + ".sse")
        rec = chat(self.base_url, self.api_key, model_id, messages, profile=self.profile,
                   ledger=self.ledger, tag=paper.name, raw_path=raw_path,
                   read_timeout=self.read_timeout, protocol=self.protocol)
        rec["paper"], rec["paper_hash"] = paper.name, paper.hash
        if self.out_dir:
            with open(os.path.join(self.out_dir, stem + ".json"), "w") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
        return rec


def list_models(base_url, api_key, timeout=30, endpoint_type=None):
    """Model ids the key can use on this route, or None if the route has no /models."""
    conn, path = _new_conn(base_url, timeout)
    try:
        conn.request("GET", path.rsplit("/chat/completions", 1)[0] + "/models",
                     headers={"Authorization": f"Bearer {api_key}", "x-api-key": api_key,
                              "anthropic-version": "2023-06-01"})
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        data = json.loads(resp.read()).get("data", [])
        if endpoint_type:
            # some gateways say which protocols each model is served over
            typed = [m for m in data if "supported_endpoint_types" in m]
            if typed:
                data = [m for m in typed if endpoint_type in m["supported_endpoint_types"]]
        return {m["id"] for m in data}
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        conn.close()
