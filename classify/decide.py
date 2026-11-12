"""Sequential decision: who gets another request, which paper, and the final answers.

Scoring rules this is built around:
  Q1  select the designated model: +20 right, -10 wrong, 0 if not selected
  Q2  every other model: +10 name, +10 version, no penalty
"""
import json
import os
import re

from candidates import CANDIDATES, OTHER
from features.extract import extract
from probes.papers import DISCRIMINATORS, FIRST_ROUND

POLICIES = {
    # name: first-round papers, stop threshold, max extra requests per id, max total requests
    "single": {"first": FIRST_ROUND["single"], "stop_p": 0.0, "max_extra": 0, "max_total": 10},
    "sequential": {"first": FIRST_ROUND["sequential"], "stop_p": 0.90, "max_extra": 2, "max_total": 30},
    "split": {"first": FIRST_ROUND["split"], "stop_p": 0.90, "max_extra": 2, "max_total": 40},
}
MIN_GAIN = 0.3


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
            self._post[key] = self.lib.posterior(self.obs(mid), exclude=self.exclude)
        return self._post[key]

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
                    extra[mid] = self.policy["max_extra"]   # nothing left that would help
                    continue
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
            select = bool(obs) and ev_select > 0 and ev_select > ev_fill
            rows.append({
                "id": mid, "select": select,
                "name": self.cands[fill]["name"] if fill else "", "version": self.cands[fill]["version"] if fill else "",
                "fill_key": fill, "top": ranked[0][1], "p_top": ranked[0][0], "p_target": p_t,
                "ev_select": ev_select, "ev_fill": ev_fill, "p_other": post.get(OTHER, 0.0),
                "top3": [(m, round(p, 4)) for p, m in ranked[:3]],
                "requests": sum(1 for _, r in self.records[mid] if r.get("counted", True)),
                "no_data": not obs,
            })
        return {"target": self.target, "rows": rows, "sent": self.sent, "notes": self.notes}


def format_answers(result, cands=None):
    cands = cands or CANDIDATES
    t = result["target"]
    out = [f"指定模型: {cands[t]['name']} {cands[t]['version']}    总请求数: {result['sent']}", ""]
    out.append(f"{'ID':<10} {'第一题':<6} {'第二题(名称/版本)':<24} {'最可能':<22} {'后验':>6} {'P(指定)':>8} {'请求':>4}  备注")
    for r in result["rows"]:
        flag = []
        if r["no_data"]:
            flag.append("无有效响应")
        if r["p_other"] > 0.5:
            flag.append("疑似库外模型")
        elif r["p_top"] < 0.6:
            flag.append("把握低")
        q1 = "选" if r["select"] else "-"
        q2 = "(第一题计分)" if r["select"] else f"{r['name']} / {r['version']}"
        out.append(f"{r['id']:<10} {q1:<6} {q2:<24} {r['top']:<22} {r['p_top']:>6.2f} {r['p_target']:>8.2f} "
                   f"{r['requests']:>4}  {'；'.join(flag)}")
    for n in result["notes"]:
        out.append(f"注意: {n}")
    return "\n".join(out)
