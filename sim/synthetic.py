"""Synthetic stand-in models, for testing the pipeline without spending API calls.

These are NOT language models. Each candidate key gets a made-up statistical
profile (skill, answer preferences, tokenizer ratio, speed); models with the same
vendor name share most of it, so siblings are hard to tell apart. Scores obtained
here show that the code works end to end. They say nothing about accuracy on
real models.
"""
import hashlib
import json
import math
import random

from candidates import CANDIDATES

_SYL = ["ka", "lo", "mi", "ren", "tu", "sa", "vel", "dor", "an", "ix", "po", "que", "bri", "ne", "shu", "ta"]
VOCAB = sorted({a + b for a in _SYL for b in _SYL})[:240]


def _rng(*parts):
    return random.Random(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest())


class SyntheticWorld:
    route = "synthetic"

    def __init__(self, seed=0, share=0.85, cands=None):
        self.seed, self.share = seed, share
        self.cands = cands or CANDIDATES
        self._prof = {}

    def keys(self):
        return list(self.cands)

    def has(self, key, paper_name):
        return True

    def profile(self, key):
        if key in self._prof:
            return self._prof[key]
        fam = self.cands[key]["name"]
        fr, mr = _rng(self.seed, "fam", fam), _rng(self.seed, "model", key)
        ratio = fr.uniform(0.24, 0.36)
        if mr.random() < 0.4:
            ratio *= mr.uniform(0.97, 1.03)
        p = {
            "fam": fam, "ratio": ratio, "skill": mr.uniform(0.15, 0.95), "think": mr.random() < 0.5,
            "tps": math.exp(mr.gauss(4.0, 0.5)), "ttft": math.exp(mr.gauss(0.0, 0.5)),
            "fence": 0.8 if fr.random() < 0.2 else 0.02,
            "digits": [max(0.2, fr.uniform(0.5, 1.5) + mr.uniform(-0.15, 0.15)) for _ in range(10)],
            "vocab": [max(0.02, fr.expovariate(1.0) + mr.uniform(-0.1, 0.1)) for _ in VOCAB],
            "ints": [max(0.05, fr.expovariate(1.0) + mr.uniform(-0.1, 0.1)) for _ in range(100)],
        }
        self._prof[key] = p
        return p

    def _pref(self, key, paper_name, item_key):
        fam = self.cands[key]["name"]
        who = fam if _rng(self.seed, "share", key, paper_name, item_key).random() < self.share else key
        return _rng(self.seed, "pref", who, paper_name, item_key).randrange(8)

    def sample(self, key, paper, cfg, rng):
        p = self.profile(key)
        shift = {"bare": 0.0, "generic": 0.05, "format": 0.08, "persona": 0.15, "disguise": 0.10}[cfg["kind"]]
        ans = {}
        for it in paper.items:
            k, kind = it["key"], it["kind"]
            pool = [f"{k}v{j}" for j in range(8)]
            if kind == "exact":
                ok = rng.random() < 1 / (1 + math.exp(-8 * (p["skill"] - 0.2 - 0.2 * it.get("level", 0))))
                ans[k] = it["answer"] if ok else "0"
            elif kind == "cat":
                if it.get("group") == "self" and cfg["kind"] == "disguise":
                    ans[k] = cfg["claim"]
                elif rng.random() < shift or rng.random() > 0.7:
                    ans[k] = rng.choice(pool)
                else:
                    ans[k] = pool[self._pref(key, paper.name, k)]
            elif kind == "digits":
                ans[k] = "".join(rng.choices("0123456789", p["digits"], k=60))
            elif kind == "ints":
                ans[k] = ", ".join(str(x) for x in rng.choices(range(1, 101), p["ints"], k=20))
            elif kind == "words":
                ans[k] = ", ".join(rng.choices(VOCAB, p["vocab"], k=12))
            elif kind == "letters":
                ans[k] = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=20))
            elif kind == "text":
                ans[k] = " ".join(rng.choices(VOCAB, p["vocab"], k=60))
            else:
                ans[k] = "none"
        text = json.dumps(ans, ensure_ascii=False)
        if rng.random() < p["fence"]:
            text = "```json\n" + text + "\n```"
        system = cfg.get("system")
        ptok = round(len(paper.prompt) * p["ratio"]) + 3 + (round(len(system) * p["ratio"]) + 2 if system else 0)
        rtok = int(math.exp(rng.gauss(6.0, 0.5))) if p["think"] else 0
        ctok = round(len(text) * p["ratio"]) + rtok
        tps = p["tps"] * math.exp(rng.gauss(0, 0.2))
        ttft = p["ttft"] * math.exp(rng.gauss(0, 0.3)) + rtok / tps
        return {
            "model_id": key, "paper": paper.name, "paper_hash": paper.hash, "status": "ok", "http_status": 200,
            "text": text, "reasoning_chars": 0,
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok,
                      "completion_tokens_details": {"reasoning_tokens": rtok}},
            "ttft": ttft, "ttft_content": ttft, "duration": ttft + (ctok - rtok) / tps,
            "finish_reason": "stop", "resp_id": "chatcmpl-" + "%024x" % rng.getrandbits(96),
            "resp_model": key, "system_fingerprint": None, "headers": {}, "error": None, "counted": True,
            "meta": {"truth": key, "route": self.route, "config": cfg["id"], "split": cfg["split"],
                     "kind": cfg["kind"], "bare": system is None, "rep": 0},
        }
