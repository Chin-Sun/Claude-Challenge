"""Run from the project root:  python3 -m unittest tests.test_all -v"""
import json
import os
import random
import socket
import tempfile
import unittest

from candidates import CANDIDATES
from classify.decide import POLICIES, AbortRun, Session
from classify.library import Library
from client.api import HttpTransport, chat
from client.ledger import BudgetExceeded, Ledger
from config import DEFAULT_WEIGHTS
from features.extract import extract, parse_answers
from probes.papers import MAIN, load_papers
from reference.configs import configs
from reference.store import to_sample
from sim.gateway import Gateway, serve
from sim.synthetic import SyntheticWorld

PAPERS = load_papers()
IDS = [f"model-{c}" for c in "abcdefghij"]
TARGET = "claude-opus-5.5"
PROFILE = {"max_tokens": 16000, "stream": True, "stream_options": {"include_usage": True}}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def synthetic_library(world, seed=1):
    rng = random.Random(seed)
    samples = [to_sample(world.sample(k, p, cfg, rng), PAPERS)
               for k in world.keys() for p in PAPERS.values() for cfg in configs("ref")[:8]]
    return Library(samples, json.loads(json.dumps(DEFAULT_WEIGHTS)))


def start_gateway(world, seed=5, fail=None):
    rng = random.Random(seed)
    truth = [TARGET] + rng.sample([k for k in world.keys() if k != TARGET], 9)
    rng.shuffle(truth)
    assign = {i: (t, rng.choice(configs("test"))) for i, t in zip(IDS, truth)}
    gw = Gateway(world, PAPERS, assign, seed=seed, fail=fail)
    port = free_port()
    srv = serve(gw, port)
    return gw, srv, f"http://127.0.0.1:{port}/v1", dict(zip(IDS, truth))


class LedgerTest(unittest.TestCase):
    def test_cap_and_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.jsonl")
            led = Ledger(path, cap=3)
            for _ in range(3):
                led.outcome(led.reserve("model-a", "t"), "ok", 200)
            with self.assertRaises(BudgetExceeded):
                led.reserve("model-a", "t")
            again = Ledger(path, cap=3)
            self.assertEqual(again.used, 3)
            self.assertEqual(again.remaining, 0)

    def test_reserve_without_outcome_still_counts(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.jsonl")
            Ledger(path, cap=5).reserve("model-a", "t")   # simulated crash before the outcome
            self.assertEqual(Ledger(path, cap=5).used, 1)


class ParserTest(unittest.TestCase):
    def test_fenced_and_truncated(self):
        keys = ["c01", "k01", "w01"]
        a, info = parse_answers('```json\n{"c01": "42", "k01": "2026-06", "w01": "hi"}\n```', keys)
        self.assertEqual(a["c01"], "42")
        self.assertTrue(info["fenced"] and info["valid_json"])
        a, info = parse_answers('{"c01": "42", "k01": "2026-06", "w01": "an unfinished sen', keys)
        self.assertEqual((a["c01"], a["k01"]), ("42", "2026-06"))
        self.assertFalse(info["valid_json"])
        self.assertEqual(a["w01"], "an unfinished sen")   # partial text is kept for the style model

    def test_model_echo_is_not_a_feature(self):
        world = SyntheticWorld(seed=2)
        rec = world.sample(TARGET, PAPERS[MAIN], configs("ref")[0], random.Random(0))
        feats = extract(rec, PAPERS[MAIN])
        self.assertFalse(any("resp_model" in k for k in feats))
        self.assertFalse(any(TARGET in json.dumps(v[2]) for v in feats.values() if v[0] == "bag"))


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = SyntheticWorld(seed=2)

    def test_ok_stream_is_parsed_and_counted(self):
        gw, srv, url, _ = start_gateway(self.world)
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(os.path.join(d, "l.jsonl"), cap=100)
            rec = HttpTransport(url, "k", ledger=led, out_dir=d).ask("model-a", PAPERS[MAIN])
            self.assertEqual(rec["status"], "ok")
            self.assertTrue(rec["text"].lstrip("`json\n").startswith("{"))
            self.assertGreater(rec["usage"]["prompt_tokens"], 0)
            self.assertEqual((led.used, gw.count), (1, 1))
            self.assertTrue(os.path.exists(os.path.join(d, f"model-a__{MAIN}.sse")))
        srv.shutdown()
        srv.server_close()

    def test_http_error_is_counted_and_never_retried(self):
        gw, srv, url, _ = start_gateway(self.world)
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(os.path.join(d, "l.jsonl"), cap=100)
            msgs = [{"role": "user", "content": PAPERS[MAIN].prompt}]
            rec = chat(url, "k", "model-a", msgs, profile={**PROFILE, "temperature": 0.5}, ledger=led)
            self.assertEqual((rec["status"], rec["http_status"]), ("http_error", 400))
            self.assertEqual((led.used, gw.count), (1, 1))
        srv.shutdown()
        srv.server_close()

    def test_connection_refused_is_not_counted(self):
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(os.path.join(d, "l.jsonl"), cap=100)
            rec = chat(f"http://127.0.0.1:{free_port()}/v1", "k", "model-a",
                       [{"role": "user", "content": "x"}], profile=PROFILE, ledger=led, connect_retries=1)
            self.assertEqual(rec["status"], "not_sent")
            self.assertFalse(rec["counted"])
            self.assertEqual(led.used, 0)


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = SyntheticWorld(seed=2)
        cls.lib = synthetic_library(cls.world)

    def test_full_run_ledger_matches_gateway_and_resume_sends_nothing(self):
        gw, srv, url, truth = start_gateway(self.world)
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(os.path.join(d, "ledger.jsonl"), cap=100)
            tr = HttpTransport(url, "k", ledger=led, out_dir=d)
            res = Session(tr, self.lib, PAPERS, IDS, TARGET, POLICIES["sequential"], run_dir=d,
                          log=lambda *_: None).run()
            self.assertEqual(led.used, gw.count)
            self.assertEqual(res["sent"], gw.count)
            self.assertGreaterEqual(gw.count, 10)
            self.assertLessEqual(gw.count, POLICIES["sequential"]["max_total"])
            right = sum(r["top"] == truth[r["id"]] for r in res["rows"])
            self.assertGreaterEqual(right, 5)   # plumbing check: far above chance (1 in 57); synthetic siblings are confusable
            before = gw.count
            led2 = Ledger(os.path.join(d, "ledger.jsonl"), cap=100)
            res2 = Session(HttpTransport(url, "k", ledger=led2, out_dir=d), self.lib, PAPERS, IDS, TARGET,
                           POLICIES["sequential"], run_dir=d, log=lambda *_: None).run()
            self.assertEqual(gw.count, before)
            self.assertEqual([r["top"] for r in res2["rows"]], [r["top"] for r in res["rows"]])
        srv.shutdown()
        srv.server_close()

    def test_rejected_first_request_aborts_after_one(self):
        gw, srv, url, _ = start_gateway(self.world, fail={"model-a": 400})
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(os.path.join(d, "ledger.jsonl"), cap=100)
            tr = HttpTransport(url, "k", ledger=led, out_dir=d)
            with self.assertRaises(AbortRun):
                Session(tr, self.lib, PAPERS, IDS, TARGET, POLICIES["sequential"], run_dir=d,
                        log=lambda *_: None).run()
            self.assertEqual((led.used, gw.count), (1, 1))
        srv.shutdown()
        srv.server_close()

    def test_cap_is_never_exceeded(self):
        gw, srv, url, _ = start_gateway(self.world)
        with tempfile.TemporaryDirectory() as d:
            led = Ledger(os.path.join(d, "ledger.jsonl"), cap=4)
            tr = HttpTransport(url, "k", ledger=led, out_dir=d)
            res = Session(tr, self.lib, PAPERS, IDS, TARGET, POLICIES["sequential"], run_dir=d,
                          log=lambda *_: None).run()
            self.assertEqual((led.used, gw.count), (4, 4))
            self.assertEqual(sum(r["no_data"] for r in res["rows"]), 6)
            self.assertFalse(any(r["select"] for r in res["rows"] if r["no_data"]))
        srv.shutdown()
        srv.server_close()

    def test_recomputed_usage_disables_tokenizer_signal(self):
        from sim.evaluate import EpisodeTransport
        rng = random.Random(3)
        truth = [TARGET] + rng.sample([k for k in self.world.keys() if k != TARGET], 9)
        assign = {i: (t, configs("test")[0]) for i, t in zip(IDS, truth)}
        tr = EpisodeTransport(self.world, assign, rng, recompute=True)
        res = Session(tr, self.lib, PAPERS, IDS, TARGET, POLICIES["single"], log=lambda *_: None).run()
        self.assertTrue(any("recomputed" in n for n in res["notes"]))
        self.assertEqual(self.lib.disabled, set())   # restored after the run


class DecisionTest(unittest.TestCase):
    def test_selects_target_only_when_expected_value_wins(self):
        class Lib:
            disabled = set()

            def __init__(self, post):
                self.post = post

            def posterior(self, obs, exclude=()):
                return self.post

        def decide(post):
            s = Session(None, Lib(post), PAPERS, ["x"], TARGET, POLICIES["single"], log=lambda *_: None)
            s.records["x"] = [(MAIN, {"status": "ok", "text": "{}", "usage": {}})]
            return s.answers()["rows"][0]

        self.assertTrue(decide({TARGET: 0.9, "claude-opus-5": 0.1})["select"])
        # 30% is below the 1/3 break-even of +20 / -10
        self.assertFalse(decide({TARGET: 0.30, "gpt-6-sol": 0.70})["select"])
        # 40% target vs a 60% sibling: filling Q2 is worth more than gambling on Q1
        row = decide({TARGET: 0.40, "claude-opus-5": 0.60})
        self.assertFalse(row["select"])
        self.assertEqual((row["name"], row["version"]), ("Claude", "Opus 5"))


if __name__ == "__main__":
    unittest.main()
