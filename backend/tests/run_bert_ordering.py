# Unit check for server.run_bert source priority: local -> Space -> None (no DB writes)
import os, sys, asyncio
sys.path.insert(0, "/app/backend")
os.environ.pop("HF_SPACE_URL", None)
import server, space

async def main():
    orig_live = server.bert_live
    r = await server.run_bert("We label emails as spam or not spam using past labelled emails.")
    print("local:", r and r["source"], r and r["label"], server.bert_source())
    server.bert_live = lambda: False
    print("unset & no local:", await server.run_bert("x"), server.bert_source())
    os.environ["HF_SPACE_URL"] = "u/s"
    space.predict = lambda t: {"label": "supervised", "confidence": 1, "latency_ms": 1, "probs": {}, "source": "hf_space"}
    r = await server.run_bert("x")
    print("space:", r["source"], server.bert_source())
    server.bert_live = orig_live
    print("local wins over space:", (await server.run_bert("x"))["source"], server.bert_source())

asyncio.run(main())
