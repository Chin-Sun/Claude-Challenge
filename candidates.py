"""Candidate model list: the answer space of the classifier.

key      internal id, also the answer identity
name     what goes in the "名称" blank
version  what goes in the "版本" blank
tier     1 = Claude (deep reference), 2 = current flagships / cheap tiers, 3 = long tail
routes   model id to send on each collection route (see endpoints.json)

ppapi ids come from the ppapi model list (no vendor prefix); anthropic ids from
the official model list. Both were read on 2026-11-12 and should be re-checked
before collecting.
"""

_C = [
    # key, name, version, tier, ppapi id, anthropic id
    ("claude-fable-5.2", "Claude", "Fable 5.2", 1, "claude-fable-5-2", "claude-fable-5-2"),
    ("claude-fable-5.1", "Claude", "Fable 5.1", 1, "claude-fable-5-1", "claude-fable-5-1"),
    ("claude-fable-5", "Claude", "Fable 5", 1, "claude-fable-5", "claude-fable-5"),
    ("claude-opus-5.5", "Claude", "Opus 5.5", 1, "claude-opus-5-5", "claude-opus-5-5"),
    ("claude-opus-5", "Claude", "Opus 5", 1, "claude-opus-5", "claude-opus-5"),
    ("claude-opus-4.8", "Claude", "Opus 4.8", 1, "claude-opus-4-8", "claude-opus-4-8"),
    ("claude-opus-4.7", "Claude", "Opus 4.7", 1, "claude-opus-4-7", "claude-opus-4-7"),
    ("claude-opus-4.6", "Claude", "Opus 4.6", 1, "claude-opus-4-6", "claude-opus-4-6"),
    ("claude-opus-4.5", "Claude", "Opus 4.5", 1, None, "claude-opus-4-5-20251101"),
    ("claude-sonnet-5.5", "Claude", "Sonnet 5.5", 1, "claude-sonnet-5-5", "claude-sonnet-5-5"),
    ("claude-sonnet-5", "Claude", "Sonnet 5", 1, "claude-sonnet-5", "claude-sonnet-5"),
    ("claude-sonnet-4.6", "Claude", "Sonnet 4.6", 1, "claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("claude-sonnet-4.5", "Claude", "Sonnet 4.5", 1, "claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
    ("claude-haiku-5.5", "Claude", "Haiku 5.5", 1, "claude-haiku-5-5", "claude-haiku-5-5"),
    ("claude-haiku-4.5", "Claude", "Haiku 4.5", 1, "claude-haiku-4-5", "claude-haiku-4-5-20251001"),

    ("gpt-6.1-sol", "GPT", "6.1 Sol", 2, "gpt-6.1-sol", None),
    ("gpt-6-sol", "GPT", "6 Sol", 2, "gpt-6-sol", None),
    ("gpt-6-luna", "GPT", "6 Luna", 2, "gpt-6-luna", None),
    ("gpt-5.6-luna", "GPT", "5.6 Luna", 2, "gpt-5.6-luna", None),
    ("gemini-4-argon", "Gemini", "4 Argon", 2, "gemini-4-argon", None),
    ("gemini-3.8-flash", "Gemini", "3.8 Flash", 2, "gemini-3.8-flash", None),
    ("gemini-3.5-flash-lite", "Gemini", "3.5 Flash Lite", 2, "gemini-3.5-flash-lite", None),
    ("gemini-3.1-pro", "Gemini", "3.1 Pro", 2, "gemini-3.1-pro-preview", None),
    ("qwen3.8-max", "Qwen", "3.8 Max", 2, "qwen3.8-max", None),
    ("qwen3.6-plus", "Qwen", "3.6 Plus", 2, "qwen3.6-plus", None),
    ("qwen3.6-flash", "Qwen", "3.6 Flash", 2, "qwen3.6-flash", None),
    ("deepseek-v4.1-flash", "DeepSeek", "V4.1 Flash", 2, "deepseek-v4.1-flash", None),
    ("deepseek-v4-pro", "DeepSeek", "V4 Pro", 2, "deepseek-v4-pro", None),
    ("doubao-seed-2.1-pro", "Doubao", "Seed 2.1 Pro", 2, "doubao-seed-2-1-pro-260915", None),
    ("doubao-seed-2.1-lite", "Doubao", "Seed 2.1 Lite", 2, "doubao-seed-2-1-lite-260915", None),
    ("glm-5.3", "GLM", "5.3", 2, "glm-5.3", None),
    ("glm-5.3-flash", "GLM", "5.3 Flash", 2, "glm-5.3-flash", None),
    ("kimi-k3.1-pro", "Kimi", "K3.1 Pro", 2, "kimi-k3.1-pro", None),
    ("grok-4.6", "Grok", "4.6", 2, "grok-4.6", None),
    ("minimax-m3", "MiniMax", "M3", 2, "MiniMax-M3", None),
    # on the contest answer sheet but not offered to our key (ids are guesses until we get access)
    ("deepseek-v4-flash", "DeepSeek", "V4 Flash", 2, "deepseek-v4-flash", None),
    ("qwen3.7-max", "Qwen", "3.7 Max", 2, "qwen3.7-max", None),
    ("qwen3.7-plus", "Qwen", "3.7 Plus", 2, "qwen3.7-plus", None),
    ("qwen3.8-plus", "Qwen", "3.8 Plus", 2, "qwen3.8-plus", None),
    # available to our ppapi key but not in the first draft of the list
    ("gpt-6-astra", "GPT", "6 Astra", 2, "gpt-6-astra", None),
    ("gpt-5.6-sol", "GPT", "5.6 Sol", 2, "gpt-5.6-sol", None),
    ("gpt-5.6-terra", "GPT", "5.6 Terra", 2, "gpt-5.6-terra", None),
    ("doubao-seed-evolving", "Doubao", "Seed Evolving", 2, "doubao-seed-evolving", None),
    ("kimi-k3.1-pro-highspeed", "Kimi", "K3.1 Pro Highspeed", 2, "kimi-k3.1-pro-highspeed", None),

    ("gpt-5.4", "GPT", "5.4", 3, "gpt-5.4", None),
    ("gpt-5.4-mini", "GPT", "5.4 Mini", 3, "gpt-5.4-mini", None),
    ("gpt-5-mini", "GPT", "5 Mini", 3, "gpt-5-mini", None),
    ("gpt-4o-mini", "GPT", "4o Mini", 3, "gpt-4o-mini", None),
    ("gemini-2.5-pro", "Gemini", "2.5 Pro", 3, "gemini-2.5-pro", None),
    ("gemini-2.5-flash", "Gemini", "2.5 Flash", 3, "gemini-2.5-flash", None),
    ("deepseek-v3.2", "DeepSeek", "V3.2", 3, "deepseek-v3.2", None),
    ("qwen3-max", "Qwen", "3 Max", 3, "qwen3-max", None),
    ("qwen-plus", "Qwen", "Plus", 3, "qwen-plus", None),
    ("doubao-seed-2.0-pro", "Doubao", "Seed 2.0 Pro", 3, "doubao-seed-2-0-pro-260215", None),
    ("kimi-k3", "Kimi", "K3", 3, "kimi-k3", None),
    ("glm-5.2", "GLM", "5.2", 3, "glm-5.2", None),
    ("grok-4.3", "Grok", "4.3", 3, "grok-4.3", None),
    ("minimax-m2.7", "MiniMax", "M2.7", 3, "MiniMax-M2.7", None),
    ("step-5-preview", "Step", "5 Preview", 3, "step-5-preview", None),
    ("hy4-preview", "Hunyuan", "4 Preview", 3, "hy4-preview", None),
    ("mimo-v2.5-pro", "MiMo", "V2.5 Pro", 3, "mimo-v2.5-pro", None),
]

OTHER = "__other__"  # open-set pseudo-candidate: "none of the above"

CANDIDATES = {
    k: {"key": k, "name": n, "version": v, "tier": t,
        "routes": {r: i for r, i in (("ppapi", p), ("anthropic", a)) if i}}
    for k, n, v, t, p, a in _C
}


# The contest pool: the designated model plus the 22 options on the answer sheet.
# Values are the option texts exactly as the sheet shows them (including its typos).
TARGET = "claude-opus-5"
POOL = {
    "claude-opus-5": "Claude Opus 5",
    "claude-opus-4.8": "Claude Opus 4.8", "claude-opus-5.5": "Claude Opus 5.5",
    "claude-sonnet-5": "Claude Sonnet 5", "claude-sonnet-5.5": "Claude Sonnet 5.5",
    "claude-fable-5.1": "Cluade Fable 5.1", "claude-fable-5.2": "Cluade Fable 5.2",
    "gpt-5.6-luna": "GPT 5.6 Luna", "gpt-5.6-sol": "GPT 5.6 Sol", "gpt-5.6-terra": "GPT 5.6 Terra",
    "gpt-6-astra": "GPT 6 Astra", "gpt-6-luna": "GPT 6 Luna", "gpt-6-sol": "GPT 6 Sol",
    "gpt-6.1-sol": "GPT 6.1 Sol",
    "deepseek-v4-flash": "DeepSeek V4 Flash", "deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "qwen3.7-max": "Qwen3.7 Max", "qwen3.7-plus": "Qwen3.7 Plus",
    "qwen3.8-max": "Qwen3.8 Max", "qwen3.8-plus": "Qwen3.8 Plus",
    "glm-5.3": "GLM 5.3", "gemini-3.8-flash": "Gemini 3.8 Flash",
}
for _k, _c in CANDIDATES.items():
    _c["pool"] = _k in POOL
    _c["label"] = POOL.get(_k, f"{_c['name']} {_c['version']}")
assert all(k in CANDIDATES for k in POOL)
# Words in a self-description that point at a family; used only for pool models we have
# no reference answers for.
FAMILY_WORDS = {"Qwen": ["qwen", "tongyi", "通义", "千问", "alibaba", "阿里"],
                "DeepSeek": ["deepseek", "深度求索"], "Claude": ["claude", "anthropic", "fable"]}


def name_of(key):
    return CANDIDATES[key]["name"] if key in CANDIDATES else "?"


def version_of(key):
    return CANDIDATES[key]["version"] if key in CANDIDATES else "?"


if __name__ == "__main__":
    import collections
    tiers = collections.Counter(c["tier"] for c in CANDIDATES.values())
    print(f"{len(CANDIDATES)} candidates; by tier: {dict(sorted(tiers.items()))}")
    for c in CANDIDATES.values():
        print(f"  T{c['tier']}  {'POOL' if c['pool'] else '    '} {c['key']:<24} {c['label']:<22} {c['routes']}")
