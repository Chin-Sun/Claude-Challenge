"""Contest-day entry point.

    export CONTEST_API_KEY=...
    python3 run_contest.py --base-url https://HOST/v1 --target claude-opus-5.5 --dry-run
    python3 run_contest.py --base-url https://HOST/v1 --target claude-opus-5.5

Re-running with the same --run continues where it stopped: requests already
paid for are read from disk, never sent again.
"""
import argparse
import json
import time

from candidates import CANDIDATES
from classify.decide import POLICIES, AbortRun, Session, format_answers
from classify.library import Library
from client.api import HttpTransport
from client.ledger import Ledger
from config import REQUEST_PROFILE, RUNS, api_key, load_weights
from probes.papers import load_papers
from reference.store import library_samples, load_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. https://host/v1")
    ap.add_argument("--key-env", default="CONTEST_API_KEY")
    ap.add_argument("--target", required=True, help="candidate key of the designated model, see candidates.py")
    ap.add_argument("--ids", default=",".join(f"model-{c}" for c in "abcdefghij"))
    ap.add_argument("--policy", default="sequential", choices=list(POLICIES))
    ap.add_argument("--stop-p", type=float, default=None)
    ap.add_argument("--max-total", type=int, default=None)
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--run", default=time.strftime("contest-%Y%m%d"))
    ap.add_argument("--lib-routes", default="", help="reference routes to build the library from (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.target not in CANDIDATES:
        raise SystemExit(f"unknown target {a.target}; run python3 candidates.py for the list")
    papers = load_papers()
    weights = load_weights()
    recs = load_records(a.lib_routes.split(",") if a.lib_routes else None)
    samples = library_samples(recs, papers, "ref")
    if not samples:
        raise SystemExit("reference library is empty; run reference.collect first")
    lib = Library(samples, weights)
    policy = dict(POLICIES[a.policy])
    if a.stop_p is not None:
        policy["stop_p"] = a.stop_p
    if a.max_total is not None:
        policy["max_total"] = a.max_total
    ids = a.ids.split(",")

    run_dir = RUNS / a.run
    ledger = Ledger(run_dir / "ledger.jsonl", cap=a.cap)
    print(f"run dir: {run_dir}")
    print(f"library: {len(lib.models)} models, {len(samples)} reference responses; "
          f"target in library: {a.target in lib.models}")
    print(f"policy: {a.policy} {policy}")
    print(f"request profile: {json.dumps(REQUEST_PROFILE)}")
    print(f"ledger: {ledger.used}/{ledger.cap} used")
    if a.target not in lib.models:
        print("WARNING: no reference data for the target model; Q1 can never be selected.")
    if a.dry_run:
        print(f"dry run: would send {len(ids) * len(policy['first'])} first-round requests to {ids}, "
              f"then up to {policy['max_total']} in total. Nothing sent.")
        return

    transport = HttpTransport(a.base_url, api_key(a.key_env), ledger=ledger, out_dir=str(run_dir))
    session = Session(transport, lib, papers, ids, a.target, policy, run_dir=str(run_dir))
    try:
        result = session.run()
    except AbortRun as e:
        print(f"\nABORTED: {e}\nledger: {ledger.used}/{ledger.cap} used")
        raise SystemExit(2)
    (run_dir / "answers.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print()
    print(format_answers(result))
    print(f"\nledger: {ledger.used}/{ledger.cap} used   answers saved to {run_dir / 'answers.json'}")


if __name__ == "__main__":
    main()
