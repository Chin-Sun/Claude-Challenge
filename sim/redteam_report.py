"""Score the classifier on each evaluation split separately.

    python3 -m sim.redteam_report                         # test + every red-team split
    python3 -m sim.redteam_report --splits rt1,rt2 --errors
    python3 -m sim.redteam_report --leave-one-out         # library = ref + the other red-team sets

No requests are sent: it replays responses collected by reference.collect.
"""
import argparse
import collections
import random
import statistics

from candidates import CANDIDATES, OTHER
from classify.decide import POLICIES, Session
from classify.library import Library
from config import load_weights
from features.extract import parse_answers
from probes.papers import MAIN, load_papers
from reference.configs import REDTEAM_SPLITS, configs
from reference.store import load_records, to_sample
from sim.evaluate import EpisodeTransport, ReplayWorld, make_episode, score

IDS = [f"model-{c}" for c in "abcdefghij"]


def usable(r):
    return r.get("status") == "ok" and (bool(r.get("text")) or r.get("finish_reason") is not None)


def evaluate_split(split, recs, lib, papers, target, episodes, seed, show_errors):
    paper = papers[MAIN]
    mine = [r for r in recs if r["meta"]["split"] == split]
    sent = len(mine)
    mine = [r for r in mine if usable(r)]
    n = top = name = claude_n = claude_ok = empty = badjson = 0
    errors, grp, shift = collections.Counter(), collections.defaultdict(float), []
    by_kind = collections.defaultdict(lambda: [0, 0])
    for r in mine:
        t = r["meta"]["truth"]
        if t not in lib.models:
            continue
        obs = [to_sample(r, papers)["feats"]]
        post = lib.posterior(obs)
        best = max(post, key=post.get)
        fill = max((m for m in post if m != OTHER), key=post.get)
        n += 1
        ok = best == t
        top += ok
        name += CANDIDATES[fill]["name"] == CANDIDATES[t]["name"]
        by_kind[r["meta"]["kind"]][0] += ok
        by_kind[r["meta"]["kind"]][1] += 1
        if CANDIDATES[t]["name"] == "Claude":
            claude_n += 1
            claude_ok += ok
        text = r.get("text") or ""
        empty += not text.strip()
        badjson += bool(text.strip()) and not parse_answers(text, paper.keys)[1]["valid_json"]
        base = lib.ptok_base.get((t, MAIN))
        pt = (r.get("usage") or {}).get("prompt_tokens")
        if base and isinstance(pt, int):
            shift.append(pt - min(base))
        if not ok:
            errors[(t, best)] += 1
            a, b = lib.loglik(t, obs)[1], lib.loglik(best, obs)[1]
            for g in set(a) | set(b):
                grp[g] += (b.get(g, 0) - a.get(g, 0)) * lib.w["groups"].get(g, 0.5)

    world = ReplayWorld(recs, papers)
    pool = [k for k in lib.models if world.has(k, MAIN, split)]
    scores, hit, miss, wrong = [], 0, 0, 0
    if target in pool and len(pool) >= 11:
        cfgs = configs(split)
        for scen in ("distinct", "adjacent", "no_target"):
            for ep in range(episodes):
                rng = random.Random(f"{seed}:{split}:{scen}:{ep}")
                truth, _ = make_episode(scen, pool, target, rng, CANDIDATES)
                assign = {i: (t, rng.choice(cfgs)) for i, t in zip(IDS, truth)}
                tr = EpisodeTransport(world, assign, random.Random(f"{seed}:{split}:{scen}:{ep}:t"))
                res = Session(tr, lib, papers, IDS, target, POLICIES["single"], log=lambda *_: None).run()
                s = score(res, dict(zip(IDS, truth)), target, CANDIDATES)
                scores.append(s["score"])
                hit += s["q1_tp"]
                miss += s["q1_fn"]
                wrong += s["q1_fp"]

    row = {"split": split, "sent": sent, "usable": n, "top1": top / max(n, 1), "name": name / max(n, 1),
           "claude": f"{claude_ok}/{claude_n}", "empty": empty, "badjson": badjson,
           "score": statistics.mean(scores) if scores else float("nan"),
           "worst": min(scores) if scores else float("nan"),
           "q1": f"{hit}/{hit + miss}", "q1_wrong": wrong,
           "ptok_shift": statistics.median(shift) if shift else float("nan")}
    detail = {"grp": dict(grp), "errors": errors, "by_kind": dict(by_kind), "pool": len(pool)}
    return row, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="")
    ap.add_argument("--target", default="claude-opus-5.5")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--errors", action="store_true", help="list confusions and signal-group damage per split")
    ap.add_argument("--leave-one-out", action="store_true",
                    help="for each red-team split, add the other red-team splits to the library")
    ap.add_argument("--no-quarantine", action="store_true")
    a = ap.parse_args()

    papers = load_papers()
    recs = [r for r in load_records(include_quarantined=a.no_quarantine)
            if r.get("paper") == MAIN and r.get("paper_hash") == papers[MAIN].hash]
    splits = a.splits.split(",") if a.splits else ["test"] + REDTEAM_SPLITS
    weights = load_weights()
    ref = [to_sample(r, papers) for r in recs if r["meta"]["split"] == "ref" and usable(r)]
    base_lib = Library(ref, weights)

    print(f"library: {len(base_lib.models)} models, {len(ref)} reference answers, "
          f"temperature {weights['temperature']}, target {a.target}"
          + ("  [leave-one-out]" if a.leave_one_out else ""))
    print(f"{'split':<7}{'sent':>5}{'usable':>7}{'top-1':>7}{'name':>6}{'Claude':>8}{'score':>7}{'worst':>6}"
          f"{'Q1 hit':>8}{'Q1 wrong':>9}{'empty':>6}{'bad json':>9}{'ptok shift':>11}")
    details = {}
    for sp in splits:
        lib = base_lib
        if a.leave_one_out and sp in REDTEAM_SPLITS:
            extra = [to_sample(r, papers) for r in recs
                     if r["meta"]["split"] in REDTEAM_SPLITS and r["meta"]["split"] != sp and usable(r)]
            lib = Library(ref + extra, weights)
        row, details[sp] = evaluate_split(sp, recs, lib, papers, a.target, a.episodes, a.seed, a.errors)
        print(f"{row['split']:<7}{row['sent']:>5}{row['usable']:>7}{row['top1']:>7.2f}{row['name']:>6.2f}"
              f"{row['claude']:>8}{row['score']:>7.1f}{row['worst']:>6.0f}{row['q1']:>8}{row['q1_wrong']:>9}"
              f"{row['empty']:>6}{row['badjson']:>9}{row['ptok_shift']:>11.0f}")
    if a.errors:
        for sp in splits:
            d = details[sp]
            print(f"\n--- {sp}: accuracy by prompt kind: "
                  + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(d["by_kind"].items())))
            print("    signal groups pushing towards the wrong answer: "
                  + ", ".join(f"{g} {v:+.0f}" for g, v in sorted(d["grp"].items(), key=lambda x: -x[1])[:6]))
            for (t, g), c in d["errors"].most_common(12):
                print(f"    {t:<26} -> {g:<26} x{c}")


if __name__ == "__main__":
    main()
