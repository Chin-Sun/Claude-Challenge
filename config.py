"""Shared paths and settings. Standard library only."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF_DATA = ROOT / "reference" / "data"
RUNS = ROOT / "runs"
FROZEN = ROOT / "probes" / "frozen"
WEIGHTS_FILE = ROOT / "weights.json"
ENDPOINTS_FILE = ROOT / "endpoints.json"

# Every request (reference, simulation, contest) uses exactly these parameters,
# so the reference library matches what the contest gateway will see.
# No temperature / top_p / n / logprobs / response_format: Claude 4.7+ rejects
# temperature < 1 on the OpenAI-compatible endpoint, and rejected requests count.
REQUEST_PROFILE = {
    # Reasoning models spend 10k-40k hidden tokens on a paper; at 16000 several returned
    # no visible answer. Every model offered to our key allows at least 64000.
    "max_tokens": 32000,
    "stream": True,
    "stream_options": {"include_usage": True},
}

# Group weights for the likelihood. Harder-to-fake signals weigh more.
DEFAULT_WEIGHTS = {
    "groups": {
        "tok": 1.0, "cap": 1.0, "know": 0.8, "beh": 0.6, "rand": 0.6,
        "style": 0.5, "fmt": 0.4, "reason": 0.5, "self": 0.15,
        "speed": 0.25, "meta": 0.25,
    },
    "temperature": 0.5,
    "tier_prior": {"1": 2.0, "2": 1.0, "3": 0.5},
    "other_prior": 0.05,
}


def load_weights():
    w = json.loads(json.dumps(DEFAULT_WEIGHTS))
    if WEIGHTS_FILE.exists():
        over = json.loads(WEIGHTS_FILE.read_text())
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(w.get(k), dict):
                w[k].update(v)
            else:
                w[k] = v
    return w


def load_endpoints():
    if not ENDPOINTS_FILE.exists():
        raise SystemExit(f"missing {ENDPOINTS_FILE}; copy endpoints.example.json and edit it")
    return json.loads(ENDPOINTS_FILE.read_text())["routes"]


def api_key(env_name):
    key = os.environ.get(env_name, "")
    if not key:
        raise SystemExit(f"environment variable {env_name} is not set")
    return key
