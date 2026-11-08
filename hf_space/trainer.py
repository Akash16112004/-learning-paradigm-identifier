import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import time
import threading
from pathlib import Path
import numpy as np
import torch
import keras
from keras.layers import TorchModuleWrapper
from transformers import AutoTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from data import LABELS, BENCHMARK

torch.set_num_threads(8)
BASE = "distilbert-base-uncased"
MAX_LEN = 64
ART = Path(__file__).parent / "artifacts"
ART.mkdir(exist_ok=True)
WEIGHTS = ART / "distilbert_keras.pt"
META = ART / "train_meta.json"

_tok = None
_model = None
_lock = threading.Lock()
STATE = {"status": "idle", "epoch": 0, "epochs": 0, "step": 0, "steps": 0, "history": [], "error": None}


def tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(BASE)
    return _tok


class DistilBertEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(BASE, attn_implementation="eager")

    def forward(self, x):
        x = x.long()
        return self.bert(input_ids=x[:, 0, :], attention_mask=x[:, 1, :]).last_hidden_state[:, 0]


class ParadigmClassifier(keras.Model):
    """Keras 3 model: DistilBERT (wrapped torch module, fully fine-tuned) + dropout + softmax head."""

    def __init__(self):
        super().__init__()
        self.encoder = TorchModuleWrapper(DistilBertEncoder())
        self.drop = keras.layers.Dropout(0.1)
        self.head = keras.layers.Dense(3, activation="softmax")

    def call(self, x, training=False):
        return self.head(self.drop(self.encoder(x), training=training))


def encode(texts):
    t = tokenizer()(list(texts), padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="np")
    return np.stack([t["input_ids"], t["attention_mask"]], 1).astype("int32")


class Progress(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        STATE["step"] += 1

    def on_epoch_end(self, epoch, logs=None):
        STATE["epoch"] = epoch + 1
        STATE["history"].append({"epoch": epoch + 1, **{k: round(float(v), 4) for k, v in (logs or {}).items()}})


def _save(model):
    torch.save({"bert": model.encoder.module.state_dict(), "head": [w for w in model.head.get_weights()]}, WEIGHTS)


def load_model():
    global _model
    if _model is None and WEIGHTS.exists():
        m = ParadigmClassifier()
        m(encode(["warmup"]))
        ck = torch.load(WEIGHTS, weights_only=False)
        m.encoder.module.load_state_dict(ck["bert"])
        m.head.set_weights(ck["head"])
        _model = m
    return _model


def is_trained():
    return WEIGHTS.exists()


def meta():
    return json.loads(META.read_text()) if META.exists() else None


def _per_domain(domains, y_true, y_pred):
    out = {}
    for d in sorted(set(domains)):
        idx = [i for i, x in enumerate(domains) if x == d]
        out[d] = {"n": len(idx), "accuracy": round(float(np.mean([y_true[i] == y_pred[i] for i in idx])), 4)}
    return out


def _train(rows, epochs, lr, batch_size):
    global _model
    try:
        texts = [r["text"] for r in rows]
        y = np.array([LABELS.index(r["label"]) for r in rows])
        doms = [r["domain"] for r in rows]
        idx = np.arange(len(rows))
        tr, va = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
        X = encode(texts)
        STATE.update(status="training", epochs=epochs, steps=int(np.ceil(len(tr) / batch_size)) * epochs)
        model = ParadigmClassifier()
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        t0 = time.time()
        model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[Progress()])
        train_secs = time.time() - t0
        STATE["status"] = "evaluating"
        pv = model.predict(X[va], verbose=0).argmax(1)
        cm = np.zeros((3, 3), int)
        for a, b in zip(y[va], pv):
            cm[a][b] += 1
        bx = encode([b["text"] for b in BENCHMARK])
        bp = model.predict(bx, verbose=0).argmax(1)
        by = [LABELS.index(b["label"]) for b in BENCHMARK]
        emb_model = model.encoder.module
        with torch.no_grad():
            embs = emb_model(torch.tensor(X)).numpy()
        pts = PCA(n_components=2, random_state=0).fit_transform(embs)
        _save(model)
        with _lock:
            _model = model
        m = {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_train": int(len(tr)), "n_val": int(len(va)), "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "train_seconds": round(train_secs, 1),
            "val_accuracy": round(float(np.mean(pv == y[va])), 4),
            "val_per_domain": _per_domain([doms[i] for i in va], list(y[va]), list(pv)),
            "confusion": cm.tolist(),
            "benchmark_accuracy": round(float(np.mean(np.array(by) == bp)), 4),
            "history": STATE["history"],
            "pca": [{"x": round(float(p[0]), 3), "y": round(float(p[1]), 3), "label": rows[i]["label"]} for i, p in enumerate(pts)],
        }
        META.write_text(json.dumps(m))
        STATE["status"] = "done"
    except Exception as e:
        STATE.update(status="error", error=str(e))


def start(rows, epochs=3, lr=3e-5, batch_size=16):
    if STATE["status"] in ("training", "evaluating", "loading"):
        return False
    STATE.update(status="loading", epoch=0, step=0, history=[], error=None)
    threading.Thread(target=_train, args=(rows, epochs, lr, batch_size), daemon=True).start()
    return True


def predict(text):
    t0 = time.perf_counter()
    model = load_model()
    if model is None:
        return None
    x = encode([text])
    with _lock, torch.no_grad():
        probs = model.predict(x, verbose=0)[0]
        out = model.encoder.module.bert(input_ids=torch.tensor(x[:, 0]).long(), attention_mask=torch.tensor(x[:, 1]).long(), output_attentions=True)
    att = out.attentions[-1][0].mean(0)[0].numpy()
    toks = tokenizer().convert_ids_to_tokens(x[0, 0])
    pairs = [(t, float(a)) for t, a, m in zip(toks, att, x[0, 1]) if m and t not in ("[CLS]", "[SEP]", "[PAD]")]
    mx = max((a for _, a in pairs), default=1) or 1
    named = {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)}
    label = max(named, key=named.get)
    return {
        "label": label, "probs": named, "confidence": named[label],
        "attention": [{"token": t, "weight": round(a / mx, 3)} for t, a in pairs],
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }
