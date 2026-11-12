"""Persistent request ledger with a hard cap.

A slot is reserved (and fsynced) right before the request bytes are sent, so a
crash can never make the ledger undercount. Anything that may have reached the
server stays counted.
"""
import json
import os
import threading
import time


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, cap=100):
        self.path = str(path)
        self.cap = cap
        self._lock = threading.Lock()
        self.entries = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(json.loads(line))

    def _append(self, entry):
        self.entries.append(entry)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @property
    def used(self):
        return sum(1 for e in self.entries if e["event"] == "reserve")

    @property
    def remaining(self):
        return self.cap - self.used

    def reserve(self, model, tag):
        with self._lock:
            if self.used >= self.cap:
                raise BudgetExceeded(f"request cap {self.cap} reached")
            seq = self.used + 1
            self._append({"event": "reserve", "seq": seq, "ts": time.time(),
                          "model": model, "tag": tag})
            return seq

    def outcome(self, seq, status, http_status=None):
        with self._lock:
            self._append({"event": "outcome", "seq": seq, "ts": time.time(),
                          "status": status, "http_status": http_status})

    def summary(self):
        by = {}
        for e in self.entries:
            if e["event"] == "reserve":
                by[e["model"]] = by.get(e["model"], 0) + 1
        return {"used": self.used, "cap": self.cap, "by_model": by}
