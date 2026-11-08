# Iteration 4: PDF report, tricky benchmark split, meta model card, Space fallback logic
import os
import io
import sys
import asyncio
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL").rstrip("/")
H = {"User-Agent": "Mozilla/5.0 pytest"}
sys.path.insert(0, "/app/backend")


@pytest.fixture(scope="module")
def pdf_bytes():
    r = requests.get(f"{BASE_URL}/api/report.pdf", headers=H, timeout=60)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/pdf")
    assert 'attachment; filename="P2_paradigm_identifier_report.pdf"' in r.headers.get("content-disposition", "")
    return r.content


# PDF report
def test_pdf_magic(pdf_bytes):
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 5000


def test_pdf_sections(pdf_bytes):
    from pypdf import PdfReader
    txt = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf_bytes)).pages)
    for s in ["Accuracy table", "Gold 20", "Tricky 10", "All", "per domain", "Hallucination", "Failure cases",
              "DistilBERT fine-tuning", "Model card", "Synthetic-data limits", "Licence"]:
        assert s in txt, s
    assert "H0" in txt or "H1" in txt  # tricky failures listed


# Eval latest with splits
def test_eval_latest_splits():
    run = requests.get(f"{BASE_URL}/api/eval/latest", headers=H, timeout=30).json()["run"]
    assert len(run["items"]) == 30
    assert set(run["splits"]) == {"core", "hard"}
    for m in ["zero_shot", "few_shot", "cot", "distilbert"]:
        assert m in run["splits"]["core"] and m in run["splits"]["hard"]
    assert set(run["distilbert_counts"]) == {"local", "hf_space", "saved"}
    assert sum(i["split"] == "hard" for i in run["items"]) == 10
    assert any(f["id"].startswith("H") for f in run["failures"])
    print({m: (run["splits"]["core"][m]["accuracy"], run["splits"]["hard"][m]["accuracy"]) for m in run["summary"]},
          run["distilbert_counts"], run["distilbert_source"])


# Meta
def test_meta():
    m = requests.get(f"{BASE_URL}/api/meta", headers=H, timeout=30).json()
    assert len(m["hard_benchmark"]) == 10
    assert all(b["id"].startswith("H") for b in m["hard_benchmark"])
    assert len(m["benchmark"]) == 20
    assert isinstance(m["model_card"], list) and len(m["model_card"]) > 3
    assert all(len(r) == 2 for r in m["model_card"])
    assert isinstance(m["licences"], list) and all(len(r) == 3 for r in m["licences"])
    assert m["bert_source"] == "local"
    assert m["hf_space_url"] == ""


# Space fallback logic (unit, no network)
def test_space_predict_none_when_unset(monkeypatch):
    import space
    monkeypatch.delenv("HF_SPACE_URL", raising=False)
    assert space.predict("x") is None
    assert space.space_page_url() == ""
    monkeypatch.setenv("HF_SPACE_URL", "user/p2")
    assert space.space_page_url() == "https://huggingface.co/spaces/user/p2"


def test_space_predict_failure_returns_none(monkeypatch):
    import space
    monkeypatch.setenv("HF_SPACE_URL", "nonexistent-user-xyz/nonexistent-space-xyz")
    space._client.cache_clear()
    assert space.predict("hello") is None
    space._client.cache_clear()


def test_space_predict_shape(monkeypatch):
    import space

    class FakeClient:
        def predict(self, text, api_name):
            assert api_name == "/classify"
            return ({}, [], "md", {"label": "supervised", "confidence": 0.9, "probs": {}, "latency_ms": 1})
    monkeypatch.setenv("HF_SPACE_URL", "u/s")
    monkeypatch.setattr(space, "_client", lambda sid: FakeClient())
    r = space.predict("hi")
    assert r["label"] == "supervised" and r["source"] == "hf_space"

    class ErrClient:
        def predict(self, text, api_name):
            return ({}, [], "Blocked", {"error": "blocked"})
    monkeypatch.setattr(space, "_client", lambda sid: ErrClient())
    assert space.predict("hi") is None
