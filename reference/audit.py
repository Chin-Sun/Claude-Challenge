"""Label audit for the reference library.

A relay can serve a request with a different backend than the one named in the
request. Such records poison the library: they carry label X but look like Y.
This tool scores every record against a library built without it (5-fold) and
lists the ones that confidently look like another model.

    python3 -m reference.audit                 # report only
    python3 -m reference.audit --write         # also write reference/quarantine.json

Quarantined records are skipped by reference.store.load_records. A flagged
record is not proof of substitution: it can also be a genuinely unusual answer.
Read the report before trusting it.
"""
import argparse
import collections
import json
import time

from candidates import OTHER
from classify.library import Library
from config import ROOT, load_weights
from probes.papers import MAIN, load_papers
from reference.store import QUARANTINE_FILE, load_records, to_sample

FOLDS = 5


def audit(records, papers, p_own_max=0.05, p_other_min=0.80):
    samples = [(r, to_sample(r, papers)) for r in records
               if r.get("status") == "ok" and r.get("text")]
    per_model = collections.defaultdict(int)
    fold = []
    for r, s in samples:
        fold.append(per_model[s["model"]] % FOLDS)
        per_model[s["model"]] += 1
    weights = load_weights()
    flagged = []
    for k in range(FOLDS):
        lib = Library([s for (r, s), f in zip(samples, fold) if f != k], weights)
        for (r, s), f in zip(samples, fold):
            if f != k:
                continue
            post = lib.posterior([s["feats"]])
            top = max(post, key=post.get)
            own = post.get(s["model"], 0.0)
            if top not in (s["model"], OTHER) and own <= p_own_max and post[top] >= p_other_min:
                flagged.append({"path": r["_path"], "label": s["model"], "looks_like": top,
                                "p_label": round(own, 4), "p_other": round(post[top], 4),
                                "config": r["meta"]["config"], "split": r["meta"]["split"],
                                "sent_at": time.strftime("%m-%d %H:%M", time.localtime(r.get("sent_at") or 0))})
    return flagged, per_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=MAIN)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    papers = load_papers()
    recs = [r for r in load_records(include_quarantined=True)
            if r.get("paper") == a.paper and r.get("paper_hash") == papers[a.paper].hash]
    flagged, per_model = audit(recs, papers)
    by = collections.Counter(f["label"] for f in flagged)
    print(f"{len(recs)} records audited, {len(flagged)} flagged")
    for m in sorted(per_model):
        if by[m]:
            looks = collections.Counter(f["looks_like"] for f in flagged if f["label"] == m)
            print(f"  {m:<26} {by[m]}/{per_model[m]} flagged, look like: {dict(looks)}")
    for f in sorted(flagged, key=lambda f: (f["label"], f["sent_at"])):
        print(f"    {f['sent_at']}  {f['label']:<24} -> {f['looks_like']:<24} "
              f"p_label={f['p_label']:<7} p_other={f['p_other']:<6} {f['split']}/{f['config']}")
    if a.write:
        old = set(json.loads(QUARANTINE_FILE.read_text())) if QUARANTINE_FILE.exists() else set()
        QUARANTINE_FILE.write_text(json.dumps(sorted(old | {f["path"] for f in flagged}), indent=1))
        print(f"wrote {QUARANTINE_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
