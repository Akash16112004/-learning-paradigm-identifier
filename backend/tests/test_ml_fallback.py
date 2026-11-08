"""Deployment-fix regression (preview, ML enabled): ml_enabled/hf_space_url flags, saved meta, distilbert in classify."""
import os
import subprocess
import sys
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL").rstrip("/")
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture(scope="module")
def s():
    x = requests.Session()
    x.headers.update({"Content-Type": "application/json"})
    return x


# Flags exposed by API
class TestFlags:
    def test_root_ml_enabled(self, s):
        j = s.get(f"{BASE_URL}/api/").json()
        assert j["ml_enabled"] is True
        assert j["trained"] is True

    def test_meta_keys(self, s):
        j = s.get(f"{BASE_URL}/api/meta").json()
        assert j["ml_enabled"] is True
        assert "hf_space_url" in j and isinstance(j["hf_space_url"], str)

    def test_train_status(self, s):
        j = s.get(f"{BASE_URL}/api/train/status").json()
        assert j["ml_enabled"] is True
        assert j["trained"] is True
        assert j["meta"]["val_accuracy"] > 0
        assert "hf_space_url" in j


# Classify still includes live DistilBERT
class TestClassifyBert:
    def test_distilbert_present(self, s):
        r = s.post(f"{BASE_URL}/api/classify", json={"text": "A robot learns to walk by trial and error, receiving rewards for forward progress."}, timeout=90)
        assert r.status_code == 200
        b = r.json()["distilbert"]
        assert b is not None
        assert b["label"] in {"supervised", "unsupervised", "reinforcement"}
        assert "attention" in b and len(b["attention"]) > 0


# Production fallback simulation (torch blocked, mocked LLM/DB)
class TestFallbackSim:
    def test_fallback(self):
        p = subprocess.run([sys.executable, os.path.join(BACKEND_DIR, "tests", "fallback_sim.py")], capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
