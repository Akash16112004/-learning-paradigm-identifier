import os
os.environ["KERAS_BACKEND"] = "torch"
import csv
import time
import gradio as gr
import trainer
import guardrail
from data import ALL_BENCHMARK, PRESETS

ROWS = list(csv.DictReader(open(os.path.join(os.path.dirname(__file__), "synthetic_200.csv"))))


def train(epochs, lr, batch_size):
    if not trainer.start(ROWS, int(epochs), float(lr), int(batch_size)):
        yield "Training is already running."
        return
    while trainer.STATE["status"] in ("loading", "training", "evaluating"):
        s = trainer.STATE
        yield f"**{s['status']}** · epoch {s['epoch']}/{s['epochs']} · step {s['step']}/{s['steps']}"
        time.sleep(3)
    if trainer.STATE["status"] == "error":
        yield f"Error: {trainer.STATE['error']}"
        return
    m = trainer.meta()
    hist = "\n".join(f"| {h['epoch']} | {h['loss']} | {h['accuracy']} | {h['val_loss']} | {h['val_accuracy']} |" for h in m["history"])
    yield (f"### Done in {m['train_seconds']} s\n- Validation accuracy (synthetic, n={m['n_val']}): **{m['val_accuracy']:.1%}**\n"
           f"- 20 hand-labelled scenarios: **{m['benchmark_accuracy']:.1%}**\n\n| epoch | loss | acc | val_loss | val_acc |\n|---|---|---|---|---|\n{hist}")


def classify(text):
    g = guardrail.check(text)
    if not g["allowed"]:
        return {}, [], "Blocked: " + " ".join(g["reasons"]), {"error": "blocked", "reasons": g["reasons"]}
    r = trainer.predict(g["text"])
    if r is None:
        return {}, [], "Model not trained yet — open the Train tab first.", {"error": "not_trained"}
    att = [(a["token"].replace("##", ""), a["weight"]) for a in r["attention"]]
    return r["probs"], att, f"{r['label']} · {r['confidence']:.1%} · {r['latency_ms']} ms", r


def benchmark():
    if not trainer.is_trained():
        return [], "Train the model first."
    rows, correct = [], 0
    for b in ALL_BENCHMARK:
        r = trainer.predict(b["text"])
        ok = r["label"] == b["label"]
        correct += ok
        rows.append([b["id"], b["domain"], b["label"], r["label"], round(r["confidence"], 3), "✓" if ok else "✗"])
    return rows, f"Accuracy on 30 hand-labelled scenarios (20 gold + 10 tricky): **{correct / len(ALL_BENCHMARK):.1%}**"


with gr.Blocks(title="P2 · DistilBERT Paradigm Classifier") as demo:
    gr.Markdown("# P2 · Learning-Paradigm Identifier — DistilBERT (Keras 3)\nFine-tunes `distilbert-base-uncased` in Keras 3 (PyTorch backend) on 200 LLM-generated scenarios and labels a scenario as supervised, unsupervised or reinforcement learning.")
    with gr.Tab("Classify"):
        inp = gr.Textbox(label="Scenario", value=PRESETS[0], lines=3)
        gr.Examples(PRESETS, inp)
        btn = gr.Button("Classify", variant="primary")
        out_label = gr.Label(label="Class probabilities")
        out_att = gr.HighlightedText(label="Last-layer [CLS] attention")
        out_txt = gr.Markdown()
        out_json = gr.JSON(label="Raw result (used by the main app API)")
        btn.click(classify, inp, [out_label, out_att, out_txt, out_json], api_name="classify")
    with gr.Tab("Train"):
        ep = gr.Slider(1, 6, value=4, step=1, label="Epochs")
        lr = gr.Dropdown([1e-5, 2e-5, 3e-5, 5e-5], value=5e-5, label="Learning rate")
        bs = gr.Dropdown([8, 16, 32], value=16, label="Batch size")
        tbtn = gr.Button("Start fine-tuning", variant="primary")
        tout = gr.Markdown()
        tbtn.click(train, [ep, lr, bs], tout)
    with gr.Tab("Benchmark"):
        bbtn = gr.Button("Score 30 hand-labelled scenarios")
        bsum = gr.Markdown()
        btab = gr.Dataframe(headers=["id", "domain", "gold", "pred", "confidence", "ok"])
        bbtn.click(benchmark, None, [btab, bsum])

demo.queue().launch()
