"""Load collected reference records and turn them into library samples."""
import json

from config import REF_DATA
from features.extract import extract


def load_records(routes=None, root=None):
    root = root or REF_DATA
    out = []
    for f in sorted(root.glob("*/*/*/*.json")):
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            continue
        meta = rec.get("meta")
        if not meta or (routes and meta["route"] not in routes):
            continue
        out.append(rec)
    return out


def to_sample(rec, papers):
    meta = rec["meta"]
    return {"model": meta["truth"], "paper": rec["paper"], "route": meta["route"],
            "bare": meta.get("bare", False), "feats": extract(rec, papers[rec["paper"]])}


def library_samples(records, papers, split="ref"):
    return [to_sample(r, papers) for r in records
            if r["meta"]["split"] == split and r.get("paper") in papers
            and r.get("paper_hash") == papers[r["paper"]].hash]


def summary(records):
    by = {}
    for r in records:
        m = r["meta"]
        k = (m["route"], m["truth"], r["paper"], m["split"])
        ok = r.get("status") == "ok" and bool(r.get("text"))
        a = by.setdefault(k, [0, 0])
        a[0] += ok
        a[1] += 1
    return by


if __name__ == "__main__":
    recs = load_records()
    print(f"{len(recs)} reference records")
    for (route, truth, paper, split), (ok, n) in sorted(summary(recs).items()):
        print(f"  {route:<10} {truth:<24} {paper:<16} {split:<5} {ok}/{n} ok")
