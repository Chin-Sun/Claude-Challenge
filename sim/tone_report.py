"""How does the target model's voice differ from the others? Uses tone-paper answers.

    python3 -m sim.tone_report                  # target = Opus 5, Messages route
    python3 -m sim.tone_report --target claude-opus-5.5

Prints, for every numeric tone feature, the target's median next to the other
models, ranked by how well that feature alone separates the target from the rest.
"""
import argparse
import collections
import math
import statistics

from candidates import CANDIDATES, TARGET
from features.extract import extract
from probes.papers import TONE, load_papers
from reference.store import load_records

READABLE = {
    "tone.len": "总长度（字符）", "tone.sent_len": "英文平均句长（词）", "tone.dash": "破折号/百词",
    "tone.semi": "分号/百词", "tone.excl": "感叹号/百词", "tone.bullet": "列表项/百词",
    "tone.bold": "加粗/百词", "tone.subhead": "小标题/百词", "tone.emoji": "表情/百词",
    "tone.question": "问号/百词", "tone.hedge": "模糊措辞/百词", "tone.paras": "每题段落数",
    "tone1.len": "安慰朋友：长度", "tone2.len": "反驳地平说：长度", "tone3.len": "讲天空：长度",
    "tone4.len": "中文雨夜：长度", "tone5.len": "婉拒邮件：长度", "tone6.len": "自评一句：长度",
    "tone4.zh_chars": "中文雨夜：汉字数", "tone4.zh_dash": "中文：破折号/百字",
    "tone4.zh_ellipsis": "中文：省略号/百字", "tone4.halfwidth": "中文：半角标点/百字",
}
CATEGORICAL = ["tone.quotes", "tone1.opener", "tone1.ends_q", "tone2.opener", "tone3.opener",
               "tone5.signoff", "tone5.subject"]


def back(name, v):
    """Undo the log transform used in features/extract.py, for display."""
    x = math.exp(v)
    return x if name.endswith(("len", "zh_chars")) or name == "tone.sent_len" else x - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--routes", default="ppapi_msg")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    paper = load_papers()[TONE]
    num = collections.defaultdict(lambda: collections.defaultdict(list))
    cat = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    n = collections.Counter()
    refused = collections.Counter()
    for r in load_records(a.routes.split(",")):
        if r.get("paper") != TONE or r.get("paper_hash") != paper.hash or r.get("status") != "ok":
            continue
        m = r["meta"]["truth"]
        if r.get("finish_reason") == "refusal":
            refused[m] += 1   # a refusal says nothing about voice; counted separately
            continue
        n[m] += 1
        for k, (ftype, _, v) in extract(r, paper).items():
            name = k.split(":", 1)[1]
            if ftype == "lognum":
                num[name][m].append(v)
            elif ftype == "bag" and name in CATEGORICAL:
                cat[name][m].update(v[0])
    t = a.target
    if not n[t]:
        raise SystemExit(f"no tone answers for {t}")
    others = sorted(m for m in n if m != t)
    print(f"tone answers used: {t} {n[t]}, others {sum(n[m] for m in others)} across {len(others)} models")
    print("refusals (excluded): " + (", ".join(f"{CANDIDATES[m].get('label', m)} {refused[m]}/{refused[m] + n[m]}"
                                               for m in sorted(refused)) or "none") + "\n")

    rows = []
    for name, by in num.items():
        if name not in READABLE or not by.get(t):
            continue
        tv = by[t]
        rest = [v for m in others for v in by.get(m, [])]
        if len(tv) < 3 or len(rest) < 3:
            continue
        sd = math.sqrt((statistics.pvariance(tv) + statistics.pvariance(rest)) / 2) or 1e-6
        d = (statistics.mean(tv) - statistics.mean(rest)) / sd
        # closest other model on this feature
        meds = {m: statistics.median(by[m]) for m in others if by.get(m)}
        near = min(meds, key=lambda m: abs(meds[m] - statistics.median(tv)))
        rows.append((abs(d), d, name, statistics.median(tv), statistics.median(rest), near, meds[near]))
    rows.sort(reverse=True)
    print(f"{'feature':<22}{'effect':>8}  {t:>14}  {'others':>8}   closest other model")
    for ad, d, name, tm, om, near, nm in rows[:a.top]:
        print(f"{READABLE[name]:<22}{d:>+8.2f}  {back(name, tm):>14.1f}  {back(name, om):>8.1f}   "
              f"{CANDIDATES[near].get('label', near)} ({back(name, nm):.1f})")
    print("\neffect = standardised difference (target minus others); |effect| > 0.8 is large.\n")

    for name in CATEGORICAL:
        if not cat[name].get(t):
            continue
        tc = cat[name][t]
        oc = sum((cat[name][m] for m in others), collections.Counter())
        top = lambda c: ", ".join(f"{k} {v / sum(c.values()):.0%}" for k, v in c.most_common(3))
        print(f"{name:<16} {t}: {top(tc):<40} others: {top(oc)}")

    print("\nper-model medians of the strongest features:")
    names = [r[2] for r in rows[:6]]
    print(f"{'model':<22}" + "".join(f"{READABLE[x][:10]:>12}" for x in names))
    for m in [t] + others:
        cells = "".join(f"{back(x, statistics.median(num[x][m])):>12.1f}" if num[x].get(m) else f"{'-':>12}"
                        for x in names)
        print(f"{CANDIDATES[m].get('label', m):<22}{cells}")


if __name__ == "__main__":
    main()
