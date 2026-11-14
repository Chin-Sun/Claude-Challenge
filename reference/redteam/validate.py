"""Format check for red-team sets. Prints counts only, never the prompt text.

    python3 -m reference.redteam.validate
"""
import json
import re
from pathlib import Path

KINDS = {"disguise", "persona", "format", "generic"}


def check(path):
    problems = []
    try:
        d = json.loads(path.read_text())
    except ValueError as e:
        return [f"not valid JSON: {e}"], {}
    ps = d.get("prompts")
    if not isinstance(d.get("persona"), str) or not isinstance(ps, list):
        return ["needs keys persona (string) and prompts (list)"], {}
    if len(ps) != 6:
        problems.append(f"needs exactly 6 prompts, has {len(ps)}")
    n_dis = n_zh = n_en = 0
    lengths = []
    for i, p in enumerate(ps):
        s = p.get("system") if isinstance(p, dict) else None
        if not isinstance(s, str) or not s.strip():
            problems.append(f"prompt {i}: empty system text")
            continue
        if p.get("kind") not in KINDS:
            problems.append(f"prompt {i}: kind must be one of {sorted(KINDS)}")
        if p.get("kind") == "disguise":
            n_dis += 1
            if "{claim}" not in s:
                problems.append(f"prompt {i}: disguise prompt must contain {{claim}}")
        if re.sub(r"\{claim\}", "", s).count("{") != re.sub(r"\{claim\}", "", s).count("}"):
            problems.append(f"prompt {i}: unbalanced braces")
        zh = len(re.findall(r"[一-鿿]", s))
        n_zh += zh > len(s) * 0.2
        n_en += zh <= len(s) * 0.2
        lengths.append(len(s))
    if n_dis < 3:
        problems.append(f"needs at least 3 disguise prompts, has {n_dis}")
    if n_zh < 1 or n_en < 1:
        problems.append("needs both Chinese and English prompts")
    if lengths and max(lengths) < 3 * min(lengths):
        problems.append("lengths too uniform: longest should be at least 3x the shortest")
    return problems, {"prompts": len(ps), "disguise": n_dis, "zh": n_zh, "en": n_en,
                      "chars": f"{min(lengths) if lengths else 0}-{max(lengths) if lengths else 0}"}


if __name__ == "__main__":
    here = Path(__file__).parent
    for n in range(1, 6):
        f = here / f"set{n}.json"
        if not f.exists():
            print(f"set{n}: MISSING")
            continue
        problems, stats = check(f)
        print(f"set{n}: {'OK' if not problems else 'FAIL'} {stats}")
        for p in problems:
            print(f"   - {p}")
