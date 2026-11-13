"""Reference fingerprint library and Bayesian posterior.

Each candidate model gets per-feature statistics pooled over many system-prompt
configurations, so disguise-induced variation widens the distributions instead
of breaking the match. A pooled background model stands in for (a) features a
candidate has no reference for and (b) the open-set candidate OTHER.
"""
import math
import statistics
from collections import Counter, defaultdict

from candidates import CANDIDATES, OTHER

BAG_ALPHA = 2.0
BAG_EFF = 8.0       # a 60-digit string counts as at most this many independent draws
TEXT_EFF = 20.0
NUM_FLOOR = 0.3     # prior std for log-scale numeric features
NUM_KAPPA = 2.0
PTOK_TOL = 2
BIN_KAPPA = 1.0
NUM_OUTLIER = 0.05   # share of the flat outlier component in numeric likelihoods
NUM_RANGE = 20.0     # width of that flat component, in log units
TEXT_MU = 300.0  # background pseudo-count for the style model


def ngrams(text):
    t = " ".join(str(text).lower().split())[:800]
    return [t[i:i + 3] for i in range(len(t) - 2)]


def _new(ftype, value):
    if ftype == "bin":
        return {"t": "bin", "k": 0, "n": 0}
    if ftype == "bag":
        return {"t": "bag", "c": Counter(), "n": 0, "m": 0, "V": value[1]}
    if ftype == "lognum":
        return {"t": "lognum", "xs": []}
    if ftype == "text":
        return {"t": "text", "c": Counter(), "n": 0}
    return None


def _add(st, ftype, value):
    if ftype == "bin":
        st["k"] += value
        st["n"] += 1
    elif ftype == "bag":
        st["c"].update(value[0])
        st["n"] += len(value[0])
        st["m"] += 1
    elif ftype == "lognum":
        st["xs"].append(value)
    elif ftype == "text":
        g = ngrams(value)
        st["c"].update(g)
        st["n"] += len(g)


def _gauss(st):
    g = st.get("g")
    if g is None or g[2] != len(st["xs"]):
        xs = st["xs"]
        mu = sum(xs) / len(xs)
        var = (sum((x - mu) ** 2 for x in xs) + NUM_KAPPA * NUM_FLOOR ** 2) / (len(xs) + NUM_KAPPA)
        g = st["g"] = (mu, var, len(xs))
    return g[0], g[1]


def _bag_p(st, tok, bg=None):
    """P(token | model), shrunk towards the pooled background distribution."""
    if bg is None or bg is st:
        return (st["c"].get(tok, 0) + BAG_ALPHA / st["V"]) / (st["n"] + BAG_ALPHA)
    prior = (bg["c"].get(tok, 0) + BAG_ALPHA / bg["V"]) / (bg["n"] + BAG_ALPHA)
    return (st["c"].get(tok, 0) + BAG_ALPHA * prior) / (st["n"] + BAG_ALPHA)


class Library:
    def __init__(self, samples, weights, candidates=None):
        """samples: dicts with model, paper, feats, bare (no system prompt), route."""
        self.cands = candidates or CANDIDATES
        self.w = weights
        self.disabled = set()
        self.stats = defaultdict(dict)
        self.bg = {}
        self.vocab = set()
        self.n_samples = Counter()
        bases = defaultdict(lambda: defaultdict(list))
        for s in samples:
            m = s["model"]
            self.n_samples[(m, s["paper"])] += 1
            for name, (ftype, group, value) in s["feats"].items():
                if ftype == "ptok":
                    if s.get("bare"):
                        bases[(m, s["paper"])][s.get("route", "")].append(value)
                    continue
                for store in (self.stats[m], self.bg):
                    if name not in store:
                        store[name] = _new(ftype, value)
                    _add(store[name], ftype, value)
                if ftype == "text":
                    self.vocab.update(ngrams(value))
        self.groups = {}
        for s in samples:
            for name, (ftype, group, _) in s["feats"].items():
                self.groups[name] = group
        # prompt_tokens baseline per (model, paper): one value per route
        # One model label can be served through several upstream channels, each adding
        # its own hidden prompt, so keep every distinct bare value seen, not one median.
        self.ptok_base = {}
        for k, routes in bases.items():
            vals = []
            for v in sorted(x for xs in routes.values() for x in xs):
                if not vals or v - vals[-1] > PTOK_TOL:
                    vals.append(v)
            self.ptok_base[k] = vals
        self.models = sorted(self.stats)
        self.papers_of = defaultdict(set)
        for (m, p), n in self.n_samples.items():
            self.papers_of[m].add(p)

    # ------------------------------------------------------------ likelihoods
    def _ll(self, st, ftype, value, bg=None):
        """Log-likelihood of one feature. Sparse per-model counts are shrunk towards the
        pooled background `bg`, so a model with few samples is never worse than OTHER
        merely for lack of data."""
        if bg is None:
            bg = st
        if ftype == "bin":
            p0 = (bg["k"] + 0.5) / (bg["n"] + 1.0)
            p = p0 if bg is st else (st["k"] + BIN_KAPPA * p0) / (st["n"] + BIN_KAPPA)
            return math.log(p if value else 1.0 - p)
        if ftype == "bag":
            toks = value[0]
            if not toks:
                return 0.0
            total = sum(math.log(_bag_p(st, t, bg)) for t in toks)
            return total * min(1.0, BAG_EFF / len(toks))
        if ftype == "lognum":
            # Gaussian mixed with a flat outlier component, so one wildly-off number
            # (e.g. speed measured over a different route) costs a bounded penalty.
            mu, var = _gauss(st)
            dens = math.exp(-(value - mu) ** 2 / (2 * var)) / math.sqrt(2 * math.pi * var)
            return math.log((1 - NUM_OUTLIER) * dens + NUM_OUTLIER / NUM_RANGE)
        if ftype == "text":
            g = ngrams(value)
            if not g:
                return 0.0
            V = len(self.vocab) + 1
            total = 0.0
            for x in g:
                p0 = (bg["c"].get(x, 0) + 0.5) / (bg["n"] + 0.5 * V)
                total += math.log(p0 if bg is st else (st["c"].get(x, 0) + TEXT_MU * p0) / (st["n"] + TEXT_MU))
            return total / len(g) * TEXT_EFF
        return 0.0

    def _ptok_single(self, model, paper, value):
        if model == OTHER:
            return math.log(1 / 3000)
        base = self.ptok_base.get((model, paper))
        if not base:
            return math.log(1 / 3000)
        best = -1e9
        for b in base:
            h = value - b   # hidden overhead added by the gateway (system prompt, template)
            if h < -PTOK_TOL:
                ll = math.log(1e-6)
            else:
                dens = 0.5 * math.exp(-max(h, 0) / 400.0) / 400.0
                if abs(h) <= PTOK_TOL:
                    dens += 0.5 / (2 * PTOK_TOL + 1)
                ll = math.log(dens)
            best = max(best, ll)
        return best

    def _ptok_diff(self, model, p1, v1, p2, v2):
        if model == OTHER:
            return math.log(1 / 3000)
        b1, b2 = self.ptok_base.get((model, p1)), self.ptok_base.get((model, p2))
        if not b1 or not b2:
            return math.log(0.02)
        d = v2 - v1
        hit = any(abs(d - (y - x)) <= PTOK_TOL for x in b1 for y in b2)
        return math.log(0.9 / (2 * PTOK_TOL + 1)) if hit else math.log(0.1 / 3000)

    def loglik(self, model, obs):
        """obs: list of feature dicts, one per answered paper. Returns (total, by_group)."""
        by = defaultdict(float)
        st_m = self.bg if model == OTHER else self.stats[model]
        ptoks = []
        for feats in obs:
            for name, (ftype, group, value) in feats.items():
                if group in self.disabled:
                    continue
                if ftype == "ptok":
                    ptoks.append((name.split(":")[0], value))
                    continue
                bg = self.bg.get(name)
                if bg is None:
                    continue
                by[group] += self._ll(st_m.get(name) or bg, ftype, value, bg)
        if ptoks and "tok" not in self.disabled:
            p1, v1 = ptoks[0]
            by["tok"] += self._ptok_single(model, p1, v1)
            for p2, v2 in ptoks[1:]:
                if p2 != p1:
                    by["tok"] += self._ptok_diff(model, p1, v1, p2, v2)
        gw = self.w["groups"]
        total = sum(gw.get(g, 0.5) * v for g, v in by.items())
        return total, dict(by)

    def prior(self, model):
        if model == OTHER:
            return self.w["other_prior"]
        tier = str(self.cands.get(model, {}).get("tier", 2))
        return self.w["tier_prior"].get(tier, 1.0)

    def posterior(self, obs, exclude=()):
        models = [m for m in self.models if m not in exclude] + [OTHER]
        T = self.w["temperature"]
        scores = {m: math.log(self.prior(m)) + T * self.loglik(m, obs)[0] for m in models}
        mx = max(scores.values())
        z = sum(math.exp(s - mx) for s in scores.values())
        return {m: math.exp(s - mx) / z for m, s in scores.items()}

    # ------------------------------------------------------------ paper selection
    def separation(self, paper, a, b, sent=()):
        """How well `paper` is expected to tell candidates a and b apart (Bhattacharyya)."""
        if a == OTHER or b == OTHER or paper not in self.papers_of[a] or paper not in self.papers_of[b]:
            return 0.0
        gw, total = self.w["groups"], 0.0
        sa, sb = self.stats[a], self.stats[b]
        for name, x in sa.items():
            if not name.startswith(paper + ":") or name not in sb:
                continue
            y, g = sb[name], self.groups.get(name, "beh")
            if g in self.disabled:
                continue
            if x["t"] == "bin":
                p, q = (x["k"] + .5) / (x["n"] + 1), (y["k"] + .5) / (y["n"] + 1)
                d = -math.log(math.sqrt(p * q) + math.sqrt((1 - p) * (1 - q)))
            elif x["t"] == "bag":
                keys = set(x["c"]) | set(y["c"])
                bc, rp, rq = 0.0, 1.0, 1.0
                for k in keys:
                    p, q = _bag_p(x, k, self.bg.get(name)), _bag_p(y, k, self.bg.get(name))
                    bc += math.sqrt(p * q)
                    rp -= p
                    rq -= q
                bc += math.sqrt(max(rp, 0) * max(rq, 0))
                eff = min(BAG_EFF, max(1.0, x["n"] / max(x["m"], 1)))
                d = -math.log(min(max(bc, 1e-9), 1.0)) * eff
            elif x["t"] == "lognum":
                (m1, v1), (m2, v2) = _gauss(x), _gauss(y)
                d = 0.25 * (m1 - m2) ** 2 / (v1 + v2) + 0.5 * math.log((v1 + v2) / (2 * math.sqrt(v1 * v2)))
            else:
                continue
            total += gw.get(g, 0.5) * d
        if "tok" not in self.disabled:
            for s in sent:
                ba, bb = self.ptok_base.get((a, paper)), self.ptok_base.get((b, paper))
                sa0, sb0 = self.ptok_base.get((a, s)), self.ptok_base.get((b, s))
                if ba and bb and sa0 and sb0:
                    da = {y - x for x in sa0 for y in ba}
                    db = {y - x for x in sb0 for y in bb}
                    if all(abs(u - v) > PTOK_TOL for u in da for v in db):
                        total += gw.get("tok", 1.0) * 3.0
                    break
        return total


def by_name(post, cands=None):
    cands = cands or CANDIDATES
    out = defaultdict(float)
    for m, p in post.items():
        out[cands[m]["name"] if m in cands else OTHER] += p
    return dict(out)
