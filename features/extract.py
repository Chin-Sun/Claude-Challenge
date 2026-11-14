"""Turn one response record into typed features.

Feature value forms (name -> (ftype, group, value)):
  bin     0/1                    exact-answer questions
  bag     (tokens, vocab_size)   categorical answers and random sequences
  lognum  float (already log)    lengths, ratios, speeds
  text    str                    free writing, scored by the style model
  ptok    int                    prompt_tokens, scored against per-model baselines
"""
import json
import math
import re

_STR = r'"((?:[^"\\]|\\.)*)"'


def parse_answers(text, keys):
    """Return (answers, info). Tolerates fences, preambles and truncated JSON."""
    info = {"fenced": "```" in text, "valid_json": False,
            "preamble": not text.lstrip().startswith(("{", "```"))}
    answers = {}
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                info["valid_json"] = True
                for k in keys:
                    if k in obj and obj[k] is not None:
                        v = obj[k]
                        answers[k] = v if isinstance(v, str) else (
                            ", ".join(map(str, v)) if isinstance(v, list) else str(v))
        except ValueError:
            pass
    if not info["valid_json"]:
        # Truncated or malformed output: recover key by key. An unterminated final
        # string (cut off by max_tokens) is kept as a partial answer.
        for k in keys:
            head = r'"' + re.escape(k) + r'"\s*:\s*'
            m = re.search(head + _STR, text) or re.search(head + r'"((?:[^"\\]|\\.)*)$', text)
            if m:
                try:
                    answers[k] = json.loads('"' + m.group(1) + '"')
                except ValueError:
                    answers[k] = m.group(1)
                continue
            m = re.search(head + r'([^\s,"}\]][^,"}\]\n]*)', text)
            if m:
                answers[k] = m.group(1).strip()
    info["n_answered"] = sum(1 for k in keys if answers.get(k, "").strip())
    return answers, info


def norm_exact(s):
    return re.sub(r"[\s,_'\"`.]+", "", str(s)).lower()


def norm_cat(s):
    s = re.sub(r"[^\w.\- ]+", " ", str(s).lower(), flags=re.UNICODE)
    return " ".join(s.split()[:4]) or "<empty>"


def _lg(x):
    return math.log(max(float(x), 1e-6))


def id_shape(rid):
    """Collapse a response id to its shape, e.g. chatcmpl-9fA3... -> chatcmpl-x29."""
    if not rid:
        return "<none>"
    m = re.match(r"^([A-Za-z_]+)[-_]?(.*)$", rid)
    if not m:
        return f"x{len(rid)}"
    return f"{m.group(1).lower()}-x{len(m.group(2))}"


def split_tone(text, n=6):
    """Split a free-form answer into its "## k" sections."""
    parts = re.split(r"(?m)^\s*#{1,3}\s*(\d)\b[^\n]*\n", text)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        k = int(parts[i])
        if 1 <= k <= n and k not in out:
            out[k] = parts[i + 1].strip()
    return out


_HEDGE = re.compile(r"\b(might|may|perhaps|possibly|probably|likely|somewhat|arguably|i think|i'd|it seems)\b", re.I)
_OPENERS = re.compile(r"^(hey|hi|oh|ah|ugh|that|i|it|so|okay|ok|wow|honestly|great|sure|well|first|the|you)\b", re.I)


def tone_features(text, put):
    """Voice and formatting habits from the free-form tone paper."""
    secs = split_tone(text)
    put("tone.sections", "lognum", "fmt", _lg(len(secs) + 1))
    put("tone.len", "lognum", "tone", _lg(len(text) + 1))
    # apostrophe and quote style differ sharply between vendors (’ vs ')
    curly = text.count("\u2019") + text.count("\u201c")
    straight = text.count("'") + text.count('"')
    put("tone.quotes", "bag", "tone",
        (["curly" if curly > straight else "straight" if straight else "none"], 3))
    en = " ".join(v for k, v in secs.items() if k != 4)
    words = max(1, len(en.split()))
    sents = max(1, len(re.findall(r"[.!?]+(\s|$)", en)))
    put("tone.sent_len", "lognum", "tone", _lg(words / sents))
    for name, pat in (("dash", r"\u2014| -- | \u2013 "), ("semi", r";"), ("excl", r"!"),
                      ("bullet", r"(?m)^\s*([-*\u2022]|\d+\.)\s"), ("bold", r"\*\*"),
                      ("subhead", r"(?m)^\s*#{1,4}\s+\D"), ("emoji", r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"),
                      ("question", r"\?")):
        put(f"tone.{name}", "lognum", "tone", _lg(1 + 100 * len(re.findall(pat, en)) / words))
    put("tone.hedge", "lognum", "tone", _lg(1 + 100 * len(_HEDGE.findall(en)) / words))
    paras = [len([p for p in re.split(r"\n\s*\n", v) if p.strip()]) for v in secs.values()]
    put("tone.paras", "lognum", "fmt", _lg(1 + (sum(paras) / len(paras) if paras else 0)))
    for k in (1, 2, 3):
        v = secs.get(k, "")
        m = _OPENERS.match(v)
        put(f"tone{k}.opener", "bag", "tone", ([(m.group(1).lower() if m else "<other>")], 20))
        put(f"tone{k}.ends_q", "bag", "tone", ([str(v.rstrip().endswith("?"))], 2))
    for k, v in secs.items():
        put(f"tone{k}", "text", "style", v)
        put(f"tone{k}.len", "lognum", "tone", _lg(len(v) + 1))
    zh = secs.get(4, "")
    if zh:
        n = max(1, len(re.findall(r"[\u4e00-\u9fff]", zh)))
        put("tone4.zh_chars", "lognum", "tone", _lg(n))
        put("tone4.zh_dash", "lognum", "tone", _lg(1 + 100 * zh.count("\u2014\u2014") / n))
        put("tone4.zh_ellipsis", "lognum", "tone", _lg(1 + 100 * zh.count("\u2026") / n))
        put("tone4.halfwidth", "lognum", "tone", _lg(1 + 100 * len(re.findall(r"[,.;:]", zh)) / n))
    mail = secs.get(5, "")
    if mail:
        last = [ln.strip() for ln in mail.splitlines() if ln.strip()]
        sign = re.sub(r"[^a-z ]", "", last[-2].lower())[:20] if len(last) >= 2 else "<none>"
        put("tone5.signoff", "bag", "tone", ([sign or "<none>"], 40))
        put("tone5.subject", "bag", "tone", ([str(bool(re.search(r"(?im)^\**subject", mail)))], 2))


def extract(rec, paper):
    f = {}
    p = paper.name

    def put(key, ftype, group, value):
        f[f"{p}:{key}"] = (ftype, group, value)

    put("status", "bag", "meta", ([rec.get("status") or "?"], 6))
    if rec.get("status") != "ok" and not rec.get("text"):
        return f
    put("empty", "bag", "fmt", ([str(not (rec.get("text") or "").strip())], 2))

    text = rec.get("text") or ""
    usage = rec.get("usage") or {}
    is_tone = any(it["kind"] == "tone" for it in paper.items)
    if is_tone:
        tone_features(text, put)
        answers, info = {}, {"fenced": False, "valid_json": False, "preamble": False,
                             "n_answered": len(split_tone(text))}
    else:
        answers, info = parse_answers(text, paper.keys)

    for it in paper.items:
        k, kind, grp = it["key"], it["kind"], it.get("group", "beh")
        a = answers.get(k, "")
        if kind == "exact":
            put(k, "bin", grp, int(bool(a) and norm_exact(a) == norm_exact(it["answer"])))
        elif kind == "cat":
            put(k, "bag", grp, ([norm_cat(a)], 50))
        elif kind == "digits":
            ds = re.findall(r"\d", a)
            put(k, "bag", grp, (ds, 10))
            put(k + ".pairs", "bag", grp, ([x + y for x, y in zip(ds, ds[1:])], 100))
            put(k + ".len", "lognum", grp, _lg(len(ds) + 1))
        elif kind == "ints":
            xs = re.findall(r"\d+", a)
            put(k, "bag", grp, (xs, 100))
            put(k + ".len", "lognum", grp, _lg(len(xs) + 1))
        elif kind == "words":
            ws = re.findall(r"[a-z]+", a.lower())
            put(k, "bag", grp, (ws, 3000))
        elif kind == "letters":
            ls = re.findall(r"[A-Za-z]", a.upper())
            put(k, "bag", grp, (ls, 26))
            put(k + ".len", "lognum", grp, _lg(len(ls) + 1))
        elif kind == "text":
            put(k, "text", "style", a)
            put(k + ".len", "lognum", "fmt", _lg(len(a) + 1))

    if not is_tone:
        put("fenced", "bag", "fmt", ([str(info["fenced"])], 2))
        put("valid_json", "bag", "fmt", ([str(info["valid_json"])], 2))
        put("preamble", "bag", "fmt", ([str(info["preamble"])], 2))
    put("answered", "lognum", "fmt", _lg(info["n_answered"] + 1))

    # Deliberately not used: rec["resp_model"]. The reference routes echo the real
    # model name there, so it would look perfectly predictive and mean nothing.
    put("finish", "bag", "meta", ([rec.get("finish_reason") or "<none>"], 8))
    put("id_shape", "bag", "meta", ([id_shape(rec.get("resp_id"))], 30))
    put("sysfp", "bag", "meta", ([str(bool(rec.get("system_fingerprint")))], 2))
    if rec.get("protocol") == "anthropic":
        # content block types (thinking shown or not) and signed thinking differ by model
        put("blocks", "bag", "reason", (["+".join(rec.get("block_types") or []) or "<none>"], 12))
        put("think_sig", "bag", "reason", ([str(bool(rec.get("thinking_signature")))], 2))

    # Which fields the gateway puts in usage differs by upstream channel.
    put("usage_keys", "bag", "meta", ([",".join(sorted(usage))[:200] or "<none>"], 30))
    put("usage_source", "bag", "meta", ([str(usage.get("usage_source"))], 10))
    put("reasoning_len", "lognum", "reason", _lg(rec.get("reasoning_chars", 0) + 1))
    details = usage.get("completion_tokens_details") or {}
    rtok = details.get("reasoning_tokens") or 0
    put("reasoning_text", "bag", "reason", ([str(rec.get("reasoning_chars", 0) > 0)], 2))
    put("reasoning_tok", "lognum", "reason", _lg(rtok + 1))

    ptok, ctok = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(ptok, int) and ptok > 0:
        put("ptok", "ptok", "tok", ptok)
    if isinstance(ctok, int) and ctok > 0:
        put("ctok", "lognum", "reason", _lg(ctok))
    if isinstance(ctok, int) and ctok > 0 and text:
        # Includes hidden reasoning tokens, so it also separates thinking models.
        put("ctok_per_char", "lognum", "tok", _lg(ctok / len(text)))
        ttft, dur = rec.get("ttft"), rec.get("duration")
        if ttft is not None and dur and dur > ttft:
            put("tps", "lognum", "speed", _lg(ctok / max(dur - ttft, 1e-3)))
    if rec.get("ttft") is not None:
        put("ttft", "lognum", "speed", _lg(rec["ttft"] + 0.05))
    return f
