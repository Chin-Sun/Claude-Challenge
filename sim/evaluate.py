"""Local mock contest: run full episodes under the contest scoring rules.

    python3 -m sim.evaluate --world synthetic                 # plumbing check, no API calls
    python3 -m sim.evaluate --world replay --split dev --calibrate
    python3 -m sim.evaluate --world replay --split test       # final estimate, run once
    python3 -m sim.evaluate --world replay --lib-routes ppapi --ep-routes anthropic   # cross-route

replay uses responses collected by reference.collect: the library is built from
the ref split, episodes are drawn from held-out dev/test configurations.
"""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict

from candidates import CANDIDATES, OTHER
from classify.decide import POLICIES, Session
from classify.library import Library
from config import WEIGHTS_FILE, load_weights
from probes.papers import MAIN, load_papers
from reference.configs import configs
from reference.store import load_records, to_sample
from sim.synthetic import SyntheticWorld

SCENARIOS = ["distinct", "dupes", "no_target", "multi_target", "adjacent", "outlib", "recompute"]


class ReplayWorld:
    def __init__(self, records, papers):
        self.idx = defaultdict(list)
        for r in records:
            # Collection-side failures (503, 504, rate limits) are not replayed as model answers.
            if r.get("paper") in papers and r.get("paper_hash") == papers[r["paper"]].hash \
                    and r.get("status") == "ok":
                self.idx[(r["meta"]["truth"], r["paper"], r["meta"]["split"])].append(r)

    def keys(self):
        return sorted({k[0] for k in self.idx})

    def has(self, key, paper_name, split=None):
        return any(k[0] == key and k[1] == paper_name and (split is None or k[2] == split) for k in self.idx)

    def sample(self, key, paper, cfg, rng):
        pool = self.idx.get((key, paper.name, cfg["split"]), [])
        same = [r for r in pool if r["meta"]["config"] == cfg["id"]]
        if not (same or pool):
            return {"status": "http_error", "http_status": 503, "text": "", "usage": {}, "counted": True,
                    "error": "no replay data", "paper": paper.name, "paper_hash": paper.hash}
        return rng.choice(same or pool)


class EpisodeTransport:
    def __init__(self, world, assign, rng, recompute=False):
        self.world, self.assign, self.rng, self.recompute = world, assign, rng, recompute
        self.count = 0

    def ask(self, mid, paper, name=None):
        key, cfg = self.assign[mid]
        rec = dict(self.world.sample(key, paper, cfg, self.rng))
        rec.pop("meta", None)
        rec["counted"] = True
        self.count += 1
        if self.recompute and rec.get("usage"):
            pt, ct = len(paper.prompt) // 4, len(rec.get("text") or "") // 4
            rec["usage"] = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        return rec


def make_episode(scenario, pool, target, rng, cands):
    others = [k for k in pool if k != target]
    exclude = ()
    if scenario == "no_target":
        truth = rng.sample(others, 10)
    elif scenario == "multi_target":
        truth = [target] * 3 + rng.sample(others, 7)
    elif scenario == "dupes":
        base = rng.sample(others, 5)
        truth = [target] + base + [rng.choice(base) for _ in range(4)]
    elif scenario == "adjacent":
        sib = [k for k in others if cands[k]["name"] == cands[target]["name"]]
        rng.shuffle(sib)
        truth = [target] + sib[:7]
        truth += rng.sample([k for k in others if k not in truth], 10 - len(truth))
    else:
        truth = [target] + rng.sample(others, 9)
        if scenario == "outlib":
            exclude = (rng.choice(truth[1:]),)
    rng.shuffle(truth)
    return truth, exclude


def score(result, truth_by_id, target, cands):
    """Contest scoring. An ID is either selected for Q1 or filled for Q2, never both."""
    s = {"score": 0, "q1_tp": 0, "q1_fp": 0, "q1_fn": 0, "name_ok": 0, "ver_ok": 0, "others": 0}
    for row in result["rows"]:
        truth = truth_by_id[row["id"]]
        is_t = truth == target
        if row["select"]:
            s["score"] += 20 if is_t else -10
            s["q1_tp" if is_t else "q1_fp"] += 1
        else:
            if is_t:
                s["q1_fn"] += 1
            else:
                s["others"] += 1
                if row["name"] == cands[truth]["name"]:
                    s["score"] += 10
                    s["name_ok"] += 1
                    if row["version"] == cands[truth]["version"]:
                        s["score"] += 10
                        s["ver_ok"] += 1
        if not is_t and row["select"]:
            s["others"] += 1
    return s


def build(a, papers):
    weights = load_weights()
    if a.world == "synthetic":
        world = SyntheticWorld(seed=a.seed)
        rng = random.Random(a.seed)
        samples = []
        for key in world.keys():
            for p in papers.values():
                for cfg in configs("ref")[:10]:
                    samples.append(to_sample(world.sample(key, p, cfg, rng), papers))
        lib_world = world
    else:
        lib_routes = a.lib_routes.split(",") if a.lib_routes else None
        ep_routes = a.ep_routes.split(",") if a.ep_routes else lib_routes
        recs = load_records(lib_routes)
        samples = [to_sample(r, papers) for r in recs if r["meta"]["split"] == "ref"
                   and r.get("paper") in papers and r.get("paper_hash") == papers[r["paper"]].hash]
        world = ReplayWorld(load_records(ep_routes), papers)
        if not samples:
            raise SystemExit("no reference data; run reference.collect first (or use --world synthetic)")
    lib = Library(samples, weights)
    return world, lib


def calibrate(world, lib, papers, split, seed, save):
    """Pick the temperature that maximises the log posterior of the truth on held-out configs."""
    rng = random.Random(seed)
    paper = papers[MAIN]
    rows = []
    for key in lib.models:
        for cfg in configs(split):
            if isinstance(world, ReplayWorld) and not world.has(key, paper.name, split):
                continue
            rec = world.sample(key, paper, cfg, rng)
            if rec.get("status") != "ok":
                continue
            obs = [to_sample({**rec, "meta": rec.get("meta") or {"truth": key, "route": "", "split": split}},
                             papers)["feats"]]
            ll = {m: (math.log(lib.prior(m)), lib.loglik(m, obs)[0]) for m in lib.models + [OTHER]}
            rows.append((key, ll))
    if not rows:
        raise SystemExit("no held-out data to calibrate on")
    best = None
    print(f"calibrating on {len(rows)} held-out responses ({split} split)")
    for T in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        nll, acc = 0.0, 0
        for key, ll in rows:
            sc = {m: lp + T * v for m, (lp, v) in ll.items()}
            mx = max(sc.values())
            z = mx + math.log(sum(math.exp(v - mx) for v in sc.values()))
            nll -= sc[key] - z
            acc += max(sc, key=sc.get) == key
        print(f"  T={T:<5} mean -log P(truth)={nll / len(rows):.3f}  top-1 accuracy={acc / len(rows):.3f}")
        if best is None or nll < best[0]:
            best = (nll, T)
    lib.w["temperature"] = best[1]
    if save:
        over = json.loads(WEIGHTS_FILE.read_text()) if WEIGHTS_FILE.exists() else {}
        over["temperature"] = best[1]
        WEIGHTS_FILE.write_text(json.dumps(over, indent=1))
        print(f"temperature set to {best[1]} (saved to {WEIGHTS_FILE.name})")
    else:
        print(f"temperature set to {best[1]} for this run only (synthetic calibration is not saved)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="synthetic", choices=["synthetic", "replay"])
    ap.add_argument("--policies", default="single,sequential")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--episodes", type=int, default=5, help="episodes per scenario")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--target", default="claude-opus-5.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lib-routes", default="")
    ap.add_argument("--ep-routes", default="")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    papers = load_papers()
    world, lib = build(a, papers)
    cands = CANDIDATES
    if a.calibrate:
        calibrate(world, lib, papers, a.split, a.seed, save=a.world == "replay")

    pool = [k for k in lib.models if k in world.keys()
            and (a.world == "synthetic" or world.has(k, MAIN, a.split))]
    if a.target not in pool or len(pool) < 11:
        raise SystemExit(f"need the target and at least 11 models with {a.split} data; have {len(pool)}")
    cfgs = configs(a.split)
    ids = [f"model-{c}" for c in "abcdefghij"]
    print(f"world={a.world} split={a.split} library models={len(lib.models)} pool={len(pool)} "
          f"temperature={lib.w['temperature']}")
    if a.world == "synthetic":
        print("NOTE: synthetic models are made-up statistical profiles. These numbers check the code, "
              "not real-world accuracy.")

    agg = {p: defaultdict(list) for p in a.policies.split(",")}
    calib = {p: defaultdict(lambda: [0, 0]) for p in agg}
    for scen in a.scenarios.split(","):
        for ep in range(a.episodes):
            rng = random.Random(f"{a.seed}:{scen}:{ep}")
            truth, exclude = make_episode("distinct" if scen == "recompute" else scen, pool, a.target, rng, cands)
            assign = {i: (t, rng.choice(cfgs)) for i, t in zip(ids, truth)}
            for pol in agg:
                tr = EpisodeTransport(world, assign, random.Random(f"{a.seed}:{scen}:{ep}:t"),
                                      recompute=scen == "recompute")
                res = Session(tr, lib, papers, ids, a.target, POLICIES[pol], exclude=exclude,
                              log=lambda *_: None).run()
                s = score(res, dict(zip(ids, truth)), a.target, cands)
                s["requests"] = tr.count
                for k, v in s.items():
                    agg[pol][(scen, k)].append(v)
                for row in res["rows"]:
                    t = dict(zip(ids, truth))[row["id"]]
                    if t in exclude:
                        continue
                    b = min(int(row["p_top"] * 5), 4)
                    calib[pol][b][0] += row["top"] == t
                    calib[pol][b][1] += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    for pol, d in agg.items():
        print(f"\n=== policy: {pol} ===")
        print(f"{'scenario':<14}{'score':>8}{'sd':>7}{'max':>6}{'requests':>10}{'name acc':>10}{'ver acc':>9}"
              f"{'Q1 hit':>8}{'Q1 wrong':>10}")
        tot_s, tot_r = [], []
        for scen in a.scenarios.split(","):
            sc, rq = d[(scen, "score")], d[(scen, "requests")]
            n_t = {"no_target": 0, "multi_target": 3}.get(scen, 1)
            others = sum(d[(scen, "others")])
            tp, fn, fp = sum(d[(scen, "q1_tp")]), sum(d[(scen, "q1_fn")]), sum(d[(scen, "q1_fp")])
            sd = statistics.pstdev(sc) if len(sc) > 1 else 0.0
            print(f"{scen:<14}{mean(sc):>8.1f}{sd:>7.1f}{20 * 10:>6}{mean(rq):>10.1f}"
                  f"{sum(d[(scen, 'name_ok')]) / max(others, 1):>10.2f}"
                  f"{sum(d[(scen, 'ver_ok')]) / max(others, 1):>9.2f}"
                  f"{(str(tp) + '/' + str(tp + fn)) if n_t else '-':>8}{fp:>10}")
            tot_s += sc
            tot_r += rq
        print(f"{'ALL':<14}{mean(tot_s):>8.1f}{statistics.pstdev(tot_s) if len(tot_s) > 1 else 0:>7.1f}"
              f"{'':>6}{mean(tot_r):>10.1f}   worst episode: {min(tot_s)}")
        print("calibration (stated confidence of top guess -> how often it was right):")
        for b in range(5):
            ok, n = calib[pol][b]
            if n:
                print(f"  {b * 20:>3}-{b * 20 + 20:<3}%  n={n:<4} right={ok / n:.2f}")


if __name__ == "__main__":
    main()
