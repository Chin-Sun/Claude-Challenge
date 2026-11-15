"""Sequential decision: who gets another request, which paper, and the final answers.

Scoring rules this is built around:
  Q1  select the designated model: +20 right, -10 wrong, 0 if not selected
  Q2  every other model: +10 name, +10 version, no penalty
"""
import json
import os
import re

import math

from candidates import CANDIDATES, FAMILY_WORDS, OTHER
from features.extract import extract, parse_answers
from probes.papers import DISCRIMINATORS, FIRST_ROUND

POLICIES = {
    # first: first-round papers; stop_p: stop once the top candidate reaches this posterior;
    # max_extra: extra requests per id; max_total: cap for the whole run;
    # retest: when no discriminator paper helps, send the main paper again (a second sample)
    "single": {"first": FIRST_ROUND["single"], "stop_p": 0.0, "max_extra": 0, "max_total": 20, "retest": False},
    "sequential": {"first": FIRST_ROUND["sequential"], "stop_p": 0.90, "max_extra": 2, "max_total": 40,
                   "retest": True},
    "split": {"first": FIRST_ROUND["split"], "stop_p": 0.90, "max_extra": 2, "max_total": 50, "retest": True},
}
MIN_GAIN = 0.3
# Each paper alone must give the target at least this posterior. The tone paper alone
# gives Opus 5 only 0.13-0.50 on held-out data and other models at most 0.15, so 0.5
# blocked every true Opus 5; 0.2 still blocks a Sonnet 5 whose main paper looked like
# an Opus 5 refusal (test05).
Q1_PAPER_MIN = 0.2


class AbortRun(RuntimeError):
    pass


def usable(rec):
    """A response we can score. A reply cut off by max_tokens before any visible text
    still counts: its token counts, reasoning trace and id shape are evidence."""
    return rec.get("status") == "ok" and (bool((rec.get("text") or "").strip())
                                          or rec.get("finish_reason") is not None)


class Session:
    def __init__(self, transport, library, papers, ids, target, policy, *, cands=None,
                 run_dir=None, exclude=(), log=print):
        self.t, self.lib, self.papers, self.ids = transport, library, papers, list(ids)
        self.target, self.policy = target, dict(policy)
        self.cands = cands or CANDIDATES
        self.run_dir, self.exclude, self.log = run_dir, tuple(exclude), log
        self.records = {i: [] for i in self.ids}
        self.sent = 0
        self.notes = []
        self._obs, self._post = {}, {}

    # ------------------------------------------------------------ sending
    def _ask(self, mid, paper_name, suffix=""):
        paper = self.papers[paper_name]
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{mid}__{paper_name}{suffix}")
        rec = None
        if self.run_dir:
            path = os.path.join(self.run_dir, stem + ".json")
            if os.path.exists(path):
                with open(path) as f:
                    old = json.load(f)
                if old.get("counted"):
                    rec = old   # resume: this request was already paid for
        if rec is None:
            rec = self.t.ask(mid, paper, name=stem)
        if rec.get("counted", True):
            self.sent += 1
        self.records[mid].append((paper_name, rec))
        self.log(f"  {mid:<10} {paper_name:<16} {rec.get('status'):<12} "
                 f"http={rec.get('http_status')} chars={len(rec.get('text') or '')}")
        return rec

    def obs(self, mid):
        key = (mid, len(self.records[mid]))
        if key not in self._obs:
            self._obs[key] = [extract(rec, self.papers[p]) for p, rec in self.records[mid] if usable(rec)]
        return self._obs[key]

    def posterior(self, mid):
        key = (mid, len(self.records[mid]), frozenset(self.lib.disabled))
        if key not in self._post:
            post = self.lib.posterior(self.obs(mid), exclude=self.exclude)
            self._post[key] = self._family_hint(mid, post)
        return self._post[key]

    def per_paper_target(self, mid):
        """P(target) from each answered paper on its own."""
        out = {}
        for pn, rec in self.records[mid]:
            if usable(rec) and pn not in out:
                f = [extract(r, self.papers[p]) for p, r in self.records[mid] if p == pn and usable(r)]
                out[pn] = self.lib.posterior(f, exclude=self.exclude).get(self.target, 0.0)
        return out

    def _family_hint(self, mid, post):
        """Pool models without reference answers ("ghosts") cannot win on fingerprints.
        If the model's own self-description names their family, shift some weight to them."""
        ghosts = getattr(self.lib, "ghosts", {})
        boost = getattr(self.lib, "w", {}).get("family_hint", 0.0) if ghosts else 0.0
        if not boost:
            return post
        said = ""
        for pn, rec in self.records[mid]:
            if usable(rec):
                ans, _ = parse_answers(rec.get("text") or "", ["s01", "s02", "s03"])
                said += " ".join(ans.values()).lower() + " "
        fams = {f for f, words in FAMILY_WORDS.items() if any(w in said for w in words)}
        if not fams:
            return post
        out = {m: p * (math.exp(boost) if m in ghosts and self.cands[m]["name"] in fams else 1.0)
               for m, p in post.items()}
        z = sum(out.values())
        return {m: p / z for m, p in out.items()}

    def _budget_left(self):
        left = self.policy["max_total"] - self.sent
        ledger = getattr(self.t, "ledger", None)
        if ledger is not None:
            left = min(left, ledger.remaining)
        return left

    # ------------------------------------------------------------ flow
    def run(self):
        saved = set(self.lib.disabled)
        try:
            return self._run()
        finally:
            self.lib.disabled = saved

    def _run(self):
        first = True
        for mid in self.ids:
            for pn in self.policy["first"]:
                if self._budget_left() <= 0:
                    self.notes.append("budget exhausted during round 1")
                    break
                rec = self._ask(mid, pn)
                if first and rec.get("status") == "http_error":
                    raise AbortRun(f"first request was rejected (HTTP {rec.get('http_status')}): "
                                   f"{rec.get('error')}. Fix the request profile before spending more.")
                first = False
                if rec.get("status") in ("timeout", "stream_error", "upstream_error") and not usable(rec) \
                        and self._budget_left() > 0:
                    self._ask(mid, pn, suffix="__retry")
        self._check_usage()

        extra = {i: 0 for i in self.ids}
        while self._budget_left() > 0:
            posts = {i: self.posterior(i) for i in self.ids}
            todo = sorted((max(p.values()), i) for i, p in posts.items()
                          if max(p.values()) < self.policy["stop_p"] and extra[i] < self.policy["max_extra"])
            progressed = False
            for _, mid in todo:
                if self._budget_left() <= 0:
                    break
                pn, gain = self.choose_paper(mid, posts[mid])
                if pn is None or gain < MIN_GAIN:
                    if not self.policy.get("retest"):
                        extra[mid] = self.policy["max_extra"]   # nothing left that would help
                        continue
                    # a second, independent sample of the main paper
                    self._ask(mid, self.policy["first"][0], suffix=f"__r{extra[mid] + 2}")
                else:
                    self._ask(mid, pn)
                extra[mid] += 1
                progressed = True
            if not progressed:
                break
        return self.answers()

    def _check_usage(self):
        """If every ID reports the same prompt_tokens, the gateway recomputes usage itself."""
        pn = self.policy["first"][0]
        vals = []
        for mid in self.ids:
            for p, rec in self.records[mid]:
                v = (rec.get("usage") or {}).get("prompt_tokens")
                if p == pn and usable(rec) and isinstance(v, int):
                    vals.append(v)
                    break
        if len(vals) >= 5 and max(vals) - min(vals) <= 1:
            self.lib.disabled = set(self.lib.disabled) | {"tok"}
            self.notes.append(f"usage looks recomputed by the gateway (prompt_tokens={vals[0]} for all); "
                              "tokenizer signals disabled")

    def choose_paper(self, mid, post):
        done = [p for p, _ in self.records[mid]]
        top = sorted(((p, m) for m, p in post.items() if m != OTHER), reverse=True)[:5]
        best, best_gain = None, 0.0
        for pn in DISCRIMINATORS:
            if pn in done or pn not in self.papers:
                continue
            gain = 0.0
            for i, (pa, a) in enumerate(top):
                for pb, b in top[i + 1:]:
                    gain += pa * pb * self.lib.separation(pn, a, b, sent=done)
            if gain > best_gain:
                best, best_gain = pn, gain
        return best, best_gain

    # ------------------------------------------------------------ answers
    def answers(self):
        rows = []
        for mid in self.ids:
            obs = self.obs(mid)
            post = self.posterior(mid)
            ranked = sorted(((p, m) for m, p in post.items()), reverse=True)
            p_t = post.get(self.target, 0.0)
            fill = next((m for _, m in ranked if m not in (OTHER, self.target)), None)
            name_p = sum(p for m, p in post.items()
                         if m in self.cands and m != self.target
                         and self.cands[m]["name"] == self.cands[fill]["name"]) if fill else 0.0
            ev_fill = 10 * name_p + 10 * post.get(fill, 0.0)
            ev_select = 20 * p_t - 10 * (1 - p_t)
            # Q1 costs 10 points when wrong, so every paper answered must point to the
            # target on its own, not just the combined posterior.
            per_paper = self.per_paper_target(mid)
            agree = all(v >= Q1_PAPER_MIN for v in per_paper.values()) if per_paper else False
            select = bool(obs) and ev_select > 0 and ev_select > ev_fill and agree
            rows.append({
                "id": mid, "select": select,
                "name": self.cands[fill]["name"] if fill else "", "version": self.cands[fill]["version"] if fill else "",
                "fill_key": fill, "label": self.cands[fill].get("label", "") if fill else "",
                "no_reference": fill in getattr(self.lib, "ghosts", {}), "top": ranked[0][1], "p_top": ranked[0][0], "p_target": p_t,
                "ev_select": ev_select, "ev_fill": ev_fill, "p_other": post.get(OTHER, 0.0),
                "top3": [(m, round(p, 4)) for p, m in ranked[:3]],
                "requests": sum(1 for _, r in self.records[mid] if r.get("counted", True)),
                "p_target_by_paper": {k: round(v, 3) for k, v in per_paper.items()},
                "no_data": not obs,
            })
        return {"target": self.target, "rows": rows, "sent": self.sent, "notes": self.notes}


def format_answers(result, cands=None):
    cands = cands or CANDIDATES
    t = result["target"]
    out = [f"指定模型: {cands[t].get('label', t)}    总请求数: {result['sent']}    "
           f"花费: ${result['sent'] * 0.01:.2f}", ""]
    out.append(f"{'ID':<10} {'第一题':<6} {'第二题（答卷选项）':<24} {'最可能':<22} {'后验':>6} {'P(指定)':>8} {'请求':>4}  备注")
    for r in result["rows"]:
        flag = []
        if r["no_data"]:
            flag.append("无有效响应")
        if r.get("no_reference"):
            flag.append("无参考数据，按家族推断")
        if r["p_other"] > 0.5:
            flag.append("和已知模型都不像")
        elif r["p_top"] < 0.6:
            flag.append("把握低")
        q1 = "选" if r["select"] else "-"
        q2 = "(第一题计分)" if r["select"] else r.get("label") or f"{r['name']} {r['version']}"
        out.append(f"{r['id']:<10} {q1:<6} {q2:<24} {r['top']:<22} {r['p_top']:>6.2f} {r['p_target']:>8.2f} "
                   f"{r['requests']:>4}  {'；'.join(flag)}")
    for n in result["notes"]:
        out.append(f"注意: {n}")
    return "\n".join(out)
