"""Collect reference responses from real models. This spends real API money.

    python3 -m reference.collect --plan                       # show what would be sent, send nothing
    python3 -m reference.collect --smoke                      # 1 request per model: does the profile get rejected?
    python3 -m reference.collect --routes ppapi               # main paper, all tiers
    python3 -m reference.collect --routes anthropic --tiers 1 --depth light
    python3 -m reference.collect --routes ppapi --tiers 1 --papers disc_capability,disc_knowledge,disc_behavior

Files land in reference/data/<route>/<model>/<paper>/<config>__<rep>.json and
existing files are skipped, so the command can be re-run after an interruption.
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from candidates import CANDIDATES
from client.api import HttpTransport, list_models
from config import REF_DATA, api_key, load_endpoints
from probes.papers import DISCRIMINATORS, load_papers
from reference.configs import configs

# configs per tier for the main paper: (ref, dev, test); bare config gets extra repeats
DEPTH = {
    "full": {1: (20, 6, 6), 2: (10, 6, 6), 3: (4, 2, 2)},
    "light": {1: (5, 2, 2), 2: (3, 1, 1), 3: (2, 1, 1)},
}
BARE_REPS = 3
OTHER_PAPER_FRACTION = 0.5   # discriminators and halves get half the configs of the main paper


def jobs(routes, tiers, paper_names, depth, only=None):
    out = []
    for key, c in CANDIDATES.items():
        if c["tier"] not in tiers or (only and key not in only):
            continue
        for route in routes:
            mid = c["routes"].get(route)
            if not mid:
                continue
            for pn in paper_names:
                if pn != "battery_v1" and c["tier"] == 3:
                    continue   # long tail: main paper only
                frac = 1.0 if pn == "battery_v1" else OTHER_PAPER_FRACTION
                n_ref, n_dev, n_test = DEPTH[depth][c["tier"]]
                for split, n in (("ref", n_ref), ("dev", n_dev), ("test", n_test)):
                    for cfg in configs(split)[: max(1, round(n * frac))]:
                        reps = BARE_REPS if cfg["kind"] == "bare" else 1
                        for rep in range(reps):
                            out.append({"key": key, "route": route, "model_id": mid, "paper": pn,
                                        "config": cfg, "rep": rep})
    return out


def path_of(job):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", job["key"])
    d = REF_DATA / job["route"] / safe / job["paper"]
    return d, f"{job['config']['id']}__{job['rep']}"


RETRYABLE = {401, 403, 408, 429, 500, 502, 503, 504}
PAPER_HASH = {}


def done(job):
    """A job is finished if it has a saved answer, or a rejection that would repeat
    (e.g. 400). Auth, rate-limit and server errors are sent again on the next run."""
    d, stem = path_of(job)
    f = d / (stem + ".json")
    if not f.exists():
        return False
    try:
        rec = json.loads(f.read_text())
    except ValueError:
        return False
    if "meta" not in rec or rec.get("paper_hash") != PAPER_HASH.get(job["paper"]):
        return False
    return rec.get("status") == "ok" or (rec.get("status") == "http_error"
                                         and rec.get("http_status") not in RETRYABLE)


def run_job(job, endpoints, papers):
    d, stem = path_of(job)
    if done(job):
        return "skip"
    ep = endpoints[job["route"]]
    t = HttpTransport(ep["base_url"], api_key(ep["key_env"]), out_dir=str(d),
                      system=job["config"]["system"])
    rec = t.ask(job["model_id"], papers[job["paper"]], name=stem)
    rec["meta"] = {"truth": job["key"], "route": job["route"], "config": job["config"]["id"],
                   "split": job["config"]["split"], "kind": job["config"]["kind"],
                   "bare": job["config"]["system"] is None, "rep": job["rep"]}
    with open(d / (stem + ".json"), "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    if rec["status"] != "ok":
        return f"{rec['status']} http={rec['http_status']} {str(rec.get('error'))[:160]}"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="ppapi")
    ap.add_argument("--tiers", default="1,2,3")
    ap.add_argument("--papers", default="battery_v1",
                    help="comma separated; discriminators: " + ",".join(DISCRIMINATORS)
                         + "; add half_a,half_b to evaluate the split policy")
    ap.add_argument("--depth", default="full", choices=list(DEPTH))
    ap.add_argument("--models", default="", help="comma separated candidate keys (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many requests")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-available", action="store_true",
                    help="with --plan: ask each route which models the key can use")
    a = ap.parse_args()

    papers = load_papers()
    PAPER_HASH.update({n: p.hash for n, p in papers.items()})
    routes = a.routes.split(",")
    tiers = {int(x) for x in a.tiers.split(",")}
    only = set(filter(None, a.models.split(",")))
    todo = jobs(routes, tiers, a.papers.split(","), a.depth, only)
    if not a.plan or a.check_available:
        # Drop models the key cannot use on a route, so they do not cost failed requests.
        endpoints = load_endpoints()
        for r in routes:
            avail = list_models(endpoints[r]["base_url"], api_key(endpoints[r]["key_env"]))
            if avail is None:
                print(f"route {r}: no model list, sending to every candidate")
                continue
            gone = sorted({j["key"] for j in todo if j["route"] == r and j["model_id"] not in avail})
            todo = [j for j in todo if j["route"] != r or j["model_id"] in avail]
            print(f"route {r}: {len({j['key'] for j in todo if j['route'] == r})} candidates available, "
                  f"{len(gone)} not offered to this key: {', '.join(gone) or '-'}")
    if a.smoke:
        seen, smoke = set(), []
        for j in todo:
            if j["paper"] == "battery_v1" and j["config"]["kind"] == "bare" and j["rep"] == 0 \
                    and (j["key"], j["route"]) not in seen:
                seen.add((j["key"], j["route"]))
                smoke.append(j)
        todo = smoke
    pending = [j for j in todo if not done(j)]
    if a.limit:
        pending = pending[: a.limit]

    by = {}
    for j in pending:
        k = (j["route"], CANDIDATES[j["key"]]["tier"], j["paper"])
        by[k] = by.get(k, 0) + 1
    print(f"{len(todo)} requests in scope, {len(todo) - len(pending) if not a.limit else '?'} already on disk, "
          f"{len(pending)} to send")
    for (route, tier, paper), n in sorted(by.items()):
        print(f"  {route:<10} tier {tier}  {paper:<16} {n}")
    if a.plan or not pending:
        return

    endpoints = load_endpoints()
    for r in routes:
        api_key(endpoints[r]["key_env"])   # fail early if a key is missing
    # Send one request alone first: a bad key or base URL should cost one request, not hundreds.
    first = run_job(pending[0], endpoints, papers)
    if "http=401" in first or "http=403" in first:
        raise SystemExit(f"authentication failed on route {pending[0]['route']}: {first}\nnothing else was sent")
    bad = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for j, res in zip(pending, ex.map(lambda j: run_job(j, endpoints, papers), pending)):
            if res not in ("ok", "skip"):
                bad += 1
                print(f"  FAILED {j['route']}/{j['key']} {j['paper']} {j['config']['id']}: {res}")
    print(f"done: {len(pending) - bad} ok, {bad} failed")
    if bad:
        print("Failed requests are saved too. A model that rejects the profile here would also "
              "reject it in the contest; check before the real run.")


if __name__ == "__main__":
    main()
