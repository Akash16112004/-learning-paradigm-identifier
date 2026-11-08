"""Simulate production host (no torch) in-process: verifies ML_ENABLED fallback paths. Uses mocked LLM + mocked DB (no writes to real DB/artifacts)."""
import asyncio
import os
import sys

sys.modules["torch"] = None  # force ImportError for trainer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

results = {}
results["ml_enabled_false"] = server.ML_ENABLED is False and server.trainer is None

bench_before = server.SAVED_BENCH.read_text()


class FakeColl:
    def __init__(self):
        self.docs = []

    async def insert_one(self, d):
        self.docs.append(dict(d))

    async def create_indexes(self, m):
        return None


class FakeDB(dict):
    def __getattr__(self, k):
        return self.setdefault(k, FakeColl())

    def __getitem__(self, k):
        return self.setdefault(k, FakeColl())


server.db = FakeDB()


def fake(label="supervised"):
    async def f(text):
        return {"label": label, "confidence": 0.9, "latency_ms": 10, "probs": {"supervised": 0.9, "unsupervised": 0.05, "reinforcement": 0.05},
                "algorithm": "Logistic Regression", "algorithm_grounded": True, "steps": [], "pipeline": [], "alternatives": [], "top_tokens": []}
    return f


server.llm.zero_shot = fake()
server.llm.few_shot = fake()
server.llm.chain_of_thought = fake()

with TestClient(server.app) as c:
    r = c.get("/api/")
    results["root_ml_false"] = r.status_code == 200 and r.json()["ml_enabled"] is False and r.json()["trained"] is False
    r = c.get("/api/meta")
    results["meta_keys"] = r.json()["ml_enabled"] is False and "hf_space_url" in r.json()
    r = c.post("/api/train", json={"epochs": 1, "lr": 3e-5, "batch_size": 16})
    results["train_503"] = r.status_code == 503
    print("train:", r.status_code, r.json())
    r = c.get("/api/train/status")
    j = r.json()
    results["status_saved_meta"] = (r.status_code == 200 and j["ml_enabled"] is False and j["meta"] is not None
                                    and "val_accuracy" in j["meta"] and j["state"]["status"] == "unavailable" and j["trained"] is False)
    print("status keys:", list(j.keys()), j["state"])
    r = c.post("/api/classify", json={"text": "A bank wants to predict loan default from labelled historical records."})
    j = r.json()
    results["classify_no_bert"] = r.status_code == 200 and j["distilbert"] is None
    print("classify:", r.status_code, j.get("distilbert"))

asyncio.run(server._eval_job())
run = server.db.eval_runs.docs[-1] if server.db.eval_runs.docs else None
results["eval_saved_source"] = bool(run) and run["distilbert_source"] == "saved" and run["distilbert_included"] is True
results["eval_distilbert_summary"] = bool(run) and "distilbert" in run["summary"]
if run:
    print("eval summary methods:", list(run["summary"].keys()), "distilbert acc:", run["summary"].get("distilbert", {}).get("accuracy"))
print("job:", server.JOBS["eval"])
results["bench_file_untouched"] = server.SAVED_BENCH.read_text() == bench_before

for k, v in results.items():
    print(("PASS" if v else "FAIL"), k)
sys.exit(0 if all(results.values()) else 1)
