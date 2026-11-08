import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, IndexModel
from starlette.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

import json  # noqa: E402
import llm  # noqa: E402
import guardrail  # noqa: E402
import space  # noqa: E402
import report  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from data import BENCHMARK, HARD_BENCHMARK, ALL_BENCHMARK, PRESETS, SYNTHETIC_LIMITS, DOMAINS, FEW_SHOT, LABELS, MODEL_CARD, LICENCES  # noqa: E402

try:  # local DistilBERT needs torch/transformers/keras; absent on the small production host
    import trainer  # noqa: E402
    ML_ENABLED = True
except ImportError:
    trainer = None
    ML_ENABLED = False

ARTIFACTS = ROOT_DIR / "artifacts"
SAVED_META = ARTIFACTS / "train_meta.json"
SAVED_BENCH = ARTIFACTS / "distilbert_benchmark.json"

client = AsyncIOMotorClient(os.environ["MONGO_URL"])
db = client[os.environ["DB_NAME"]]
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INDEXES = {
    "synthetic": [IndexModel([("domain", ASCENDING)], name="domain")],
    "classifications": [IndexModel([("created_at", DESCENDING)], name="created_desc")],
    "eval_runs": [IndexModel([("created_at", DESCENDING)], name="created_desc")],
}

app = FastAPI(title="P2 Learning-Paradigm Identifier")
api = APIRouter(prefix="/api")
JOBS = {"synthetic": {"status": "idle", "error": None}, "eval": {"status": "idle", "done": 0, "total": 0, "error": None}}


def now():
    return datetime.now(timezone.utc).isoformat()


class TextIn(BaseModel):
    text: str


class IllustrateIn(BaseModel):
    text: str
    paradigm: str
    algorithm: str = ""
    pipeline: List[str] = Field(default_factory=list)


class TrainIn(BaseModel):
    epochs: int = Field(3, ge=1, le=8)
    lr: float = Field(3e-5, gt=0, le=1e-3)
    batch_size: int = Field(16, ge=4, le=32)


def bert_live():
    return ML_ENABLED and trainer.is_trained()


def bert_source():
    return "local" if bert_live() else ("hf_space" if space.space_id() else None)


async def run_bert(text):
    if bert_live():
        r = await asyncio.to_thread(trainer.predict, text)
        return {**r, "source": "local"}
    if space.space_id():
        return await asyncio.to_thread(space.predict, text)
    return None


def saved_json(path):
    return json.loads(path.read_text()) if path.exists() else None


@api.get("/")
async def root():
    return {"message": "Learning-Paradigm Identifier API", "llm": llm.MODEL, "ml_enabled": ML_ENABLED, "trained": bert_live(), "bert_source": bert_source()}


@api.get("/meta")
async def meta():
    return {
        "presets": PRESETS, "benchmark": BENCHMARK, "hard_benchmark": HARD_BENCHMARK, "limits": SYNTHETIC_LIMITS, "domains": DOMAINS,
        "few_shot": [{"text": t, "answer": a} for t, a in FEW_SHOT],
        "prompts": {"zero_shot": llm.LETTER_SYSTEM, "cot": llm.COT_SYSTEM},
        "llm_model": llm.MODEL, "image_model": llm.IMAGE_MODEL, "bert_model": "distilbert-base-uncased",
        "ml_enabled": ML_ENABLED, "hf_space_url": space.space_page_url(), "bert_source": bert_source(),
        "model_card": MODEL_CARD, "licences": LICENCES,
    }


@api.post("/guardrail")
async def guard(body: TextIn):
    return guardrail.check(body.text)


@api.post("/classify")
async def classify(body: TextIn):
    g = guardrail.check(body.text)
    if not g["allowed"]:
        return {"blocked": True, "guardrail": g}
    text = g["text"]
    try:
        cot, zs, fs, bert = await asyncio.gather(llm.chain_of_thought(text), llm.zero_shot(text), llm.few_shot(text), run_bert(text))
    except Exception as e:
        logger.exception("classify failed")
        raise HTTPException(502, f"Model call failed: {e}")
    off_topic = cot["label"] == "not_ml"
    doc = {"id": str(uuid.uuid4()), "text": text, "cot_label": cot["label"], "confidence": cot["confidence"],
           "algorithm": cot["algorithm"], "zero_shot": zs["label"], "few_shot": fs["label"],
           "distilbert": bert["label"] if bert else None, "created_at": now()}
    await db.classifications.insert_one(doc)
    return {"blocked": False, "off_topic": off_topic, "guardrail": g, "cot": cot, "zero_shot": zs, "few_shot": fs, "distilbert": bert, "id": doc["id"]}


@api.post("/illustrate")
async def illustrate(body: IllustrateIn):
    try:
        img = await llm.illustrate(body.text, body.paradigm, body.algorithm, body.pipeline)
    except Exception as e:
        logger.exception("illustrate failed")
        raise HTTPException(502, f"Image generation failed: {e}")
    if not img:
        raise HTTPException(502, "No image returned")
    return img


@api.get("/history")
async def history():
    return await db.classifications.find({}, {"_id": 0}).sort("created_at", -1).to_list(20)


async def _gen_job():
    try:
        rows = await llm.generate_all(25)
        if len(rows) < 50:
            raise RuntimeError(f"Only {len(rows)} scenarios generated")
        stamp = now()
        for r in rows:
            r.update(id=str(uuid.uuid4()), created_at=stamp)
        await db.synthetic.delete_many({})
        await db.synthetic.insert_many(rows)
        JOBS["synthetic"].update(status="done", count=len(rows))
    except Exception as e:
        logger.exception("synthetic generation failed")
        JOBS["synthetic"].update(status="error", error=str(e))


@api.post("/synthetic/generate")
async def gen_synthetic():
    if JOBS["synthetic"]["status"] == "running":
        return JOBS["synthetic"]
    JOBS["synthetic"] = {"status": "running", "error": None, "started_at": time.time()}
    asyncio.create_task(_gen_job())
    return JOBS["synthetic"]


@api.get("/synthetic")
async def list_synthetic():
    rows = await db.synthetic.find({}, {"_id": 0}).to_list(1000)
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in LABELS}
    by_domain = {d: sum(r["domain"] == d for r in rows) for d in DOMAINS}
    return {"rows": rows, "counts": counts, "by_domain": by_domain, "job": JOBS["synthetic"]}


@api.post("/train")
async def train(body: TrainIn):
    if not ML_ENABLED:
        raise HTTPException(503, "Live training is not available on this host; it runs in the preview or the Hugging Face Space")
    rows = await db.synthetic.find({}, {"_id": 0}).to_list(1000)
    if len(rows) < 30:
        raise HTTPException(400, "Generate the synthetic dataset first")
    if not trainer.start(rows, body.epochs, body.lr, body.batch_size):
        raise HTTPException(409, "Training already running")
    return trainer.STATE


@api.get("/train/status")
async def train_status():
    url = space.space_page_url()
    if not ML_ENABLED:
        return {"state": {"status": "unavailable", "history": []}, "meta": saved_json(SAVED_META), "trained": False,
                "ml_enabled": False, "hf_space_url": url}
    return {"state": trainer.STATE, "meta": trainer.meta(), "trained": trainer.is_trained(), "ml_enabled": True, "hf_space_url": url}


def _summary(items, methods):
    s = {}
    for m in methods:
        rs = [i[m] for i in items if i.get(m)]
        if not rs:
            continue
        per_dom = {}
        for d in DOMAINS:
            sub = [i for i in items if i["domain"] == d and i.get(m)]
            if sub:
                per_dom[d] = round(sum(i[m]["label"] == i["label"] for i in sub) / len(sub), 4)
        accs = list(per_dom.values())
        s[m] = {
            "accuracy": round(sum(i[m]["label"] == i["label"] for i in items if i.get(m)) / len(rs), 4),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in rs) / len(rs)),
            "mean_confidence": round(sum(r["confidence"] for r in rs) / len(rs), 4),
            "per_domain": per_dom,
            "domain_gap": round(max(accs) - min(accs), 4) if accs else 0,
        }
    return s


async def _eval_job():
    sem = asyncio.Semaphore(5)
    source = bert_source()
    saved = saved_json(SAVED_BENCH) or {}
    used = {"local": 0, "hf_space": 0, "saved": 0}

    async def one(b):
        try:
            async with sem:
                zs, fs, cot = await asyncio.gather(llm.zero_shot(b["text"]), llm.few_shot(b["text"]), llm.chain_of_thought(b["text"]))
            bert = await run_bert(b["text"]) if source else None
            if bert:
                used[bert["source"]] += 1
            elif saved.get(b["id"]):
                bert = saved[b["id"]]
                used["saved"] += 1
        except Exception:
            logger.exception("eval item %s failed", b["id"])
            return None
        JOBS["eval"]["done"] += 1
        slim = lambda r: r and {k: r[k] for k in ("label", "confidence", "latency_ms", "probs")}
        return {**b, "split": "hard" if b["id"].startswith("H") else "core", "zero_shot": slim(zs), "few_shot": slim(fs),
                "cot": {**slim(cot), "algorithm": cot["algorithm"], "algorithm_grounded": cot["algorithm_grounded"]}, "distilbert": slim(bert)}

    try:
        items = [i for i in await asyncio.gather(*[one(b) for b in ALL_BENCHMARK]) if i]
        if not items:
            raise RuntimeError("All benchmark items failed")
        methods = ["zero_shot", "few_shot", "cot", "distilbert"]
        grounded = [i["cot"]["algorithm_grounded"] for i in items]
        failures = [{"id": i["id"], "text": i["text"], "gold": i["label"], "domain": i["domain"], "split": i["split"],
                     "wrong": {m: i[m]["label"] for m in methods if i.get(m) and i[m]["label"] != i["label"]}}
                    for i in items if any(i.get(m) and i[m]["label"] != i["label"] for m in methods)]
        main_src = max(used, key=used.get) if any(used.values()) else None
        run = {"id": str(uuid.uuid4()), "created_at": now(), "items": items, "summary": _summary(items, methods),
               "splits": {sp: _summary([i for i in items if i["split"] == sp], methods) for sp in ("core", "hard")},
               "hallucination": {"algorithm_grounded_rate": round(sum(grounded) / len(grounded), 4),
                                 "ungrounded": [{"id": i["id"], "algorithm": i["cot"]["algorithm"], "paradigm": i["cot"]["label"]} for i in items if not i["cot"]["algorithm_grounded"]]},
               "failures": failures, "distilbert_included": main_src is not None, "distilbert_source": main_src,
               "distilbert_counts": used, "llm_model": llm.MODEL}
        await db.eval_runs.insert_one(run)
        if used["local"] == len(items):
            SAVED_BENCH.write_text(json.dumps({i["id"]: i["distilbert"] for i in items}))
        JOBS["eval"]["status"] = "done"
    except Exception as e:
        logger.exception("eval failed")
        JOBS["eval"].update(status="error", error=str(e))


@api.post("/eval/run")
async def eval_run():
    if JOBS["eval"]["status"] == "running":
        return JOBS["eval"]
    JOBS["eval"] = {"status": "running", "done": 0, "total": len(ALL_BENCHMARK), "error": None}
    asyncio.create_task(_eval_job())
    return JOBS["eval"]


@api.get("/eval/latest")
async def eval_latest():
    run = await db.eval_runs.find_one({}, {"_id": 0}, sort=[("created_at", -1)])
    return {"run": run, "job": JOBS["eval"]}


@api.get("/report.pdf")
async def report_pdf():
    run = await db.eval_runs.find_one({}, {"_id": 0}, sort=[("created_at", -1)])
    m = trainer.meta() if ML_ENABLED else saved_json(SAVED_META)
    pdf = await asyncio.to_thread(report.build, run, m, {"llm": llm.MODEL, "image": llm.IMAGE_MODEL})
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="P2_paradigm_identifier_report.pdf"'})


app.include_router(api)
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    for coll, models in INDEXES.items():
        try:
            await db[coll].create_indexes(models)
        except Exception as exc:
            logger.error("index %s: %s", coll, exc)


@app.on_event("shutdown")
async def shutdown():
    client.close()
