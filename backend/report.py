from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from data import MODEL_CARD, LICENCES, SYNTHETIC_LIMITS, LABELS

NAMES = {"zero_shot": "Zero-shot LLM", "few_shot": "Few-shot LLM", "cot": "Chain-of-thought LLM", "distilbert": "DistilBERT (Keras)"}
SS = getSampleStyleSheet()
H1, H2, BODY = SS["Title"], SS["Heading2"], SS["BodyText"]
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10)


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def table(rows, widths=None, header=True):
    rows = [[Paragraph(str(c), SMALL) for c in r] for r in rows]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEFF5")))
    t.setStyle(TableStyle(style))
    return t


def build(run, meta, info):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    s = []
    s.append(Paragraph("P2 - Learning-Paradigm Identifier: Evaluation Report", H1))
    s.append(Paragraph(f"LLM: {info['llm']} | Classifier: distilbert-base-uncased (Keras 3) | Image: {info['image']} | Run: {run['created_at'][:19] if run else '-'}", SMALL))
    s.append(Spacer(1, 8))

    if run:
        splits = run.get("splits", {})
        s.append(Paragraph("1. Accuracy table", H2))
        rows = [["Method", "Gold 20", "Tricky 10", "All", "Mean confidence", "Mean latency"]]
        for m, v in run["summary"].items():
            rows.append([NAMES[m], pct(splits.get("core", {}).get(m, {}).get("accuracy")), pct(splits.get("hard", {}).get(m, {}).get("accuracy")),
                         pct(v["accuracy"]), pct(v["mean_confidence"]), f"{v['mean_latency_ms']} ms"])
        s.append(table(rows))
        if run.get("distilbert_source") == "saved":
            s.append(Paragraph("DistilBERT rows use saved predictions from the last training run.", SMALL))
        s.append(Spacer(1, 6))

        s.append(Paragraph("2. Accuracy per domain (bias probe)", H2))
        methods = list(run["summary"].keys())
        domains = sorted({d for v in run["summary"].values() for d in v["per_domain"]})
        rows = [["Domain"] + [NAMES[m] for m in methods]]
        for d in domains:
            rows.append([d] + [pct(run["summary"][m]["per_domain"].get(d)) for m in methods])
        rows.append(["Max gap"] + [pct(run["summary"][m]["domain_gap"]) for m in methods])
        s.append(table(rows))

        s.append(Paragraph("3. Hallucination test", H2))
        h = run["hallucination"]
        s.append(Paragraph(f"Algorithm grounding rate (CoT suggestion is a real algorithm for the predicted paradigm): <b>{pct(h['algorithm_grounded_rate'])}</b>", BODY))
        for u in h["ungrounded"]:
            s.append(Paragraph(f"- {u['id']}: '{u['algorithm']}' for {u['paradigm']}", SMALL))

        s.append(Paragraph(f"4. Failure cases ({len(run['failures'])})", H2))
        rows = [["ID", "Scenario", "Gold", "Wrong predictions"]]
        for f in run["failures"]:
            rows.append([f["id"], f["text"], f["gold"], ", ".join(f"{NAMES[m]}: {lab}" for m, lab in f["wrong"].items())])
        s.append(table(rows, widths=[12 * mm, 95 * mm, 22 * mm, 49 * mm]))
    else:
        s.append(Paragraph("No benchmark run yet - run it on the Evaluation page.", BODY))

    s.append(PageBreak())
    if meta:
        s.append(Paragraph("5. DistilBERT fine-tuning (Keras 3)", H2))
        s.append(Paragraph(f"{meta['n_train']} train / {meta['n_val']} validation synthetic examples, {meta['epochs']} epochs, lr {meta['lr']}, batch {meta['batch_size']}, "
                           f"{meta['train_seconds']} s on CPU. Validation accuracy <b>{pct(meta['val_accuracy'])}</b>; gold-20 accuracy <b>{pct(meta['benchmark_accuracy'])}</b>.", BODY))
        rows = [["gold \\ pred"] + LABELS] + [[LABELS[i]] + row for i, row in enumerate(meta["confusion"])]
        s.append(table(rows))
        rows = [["Epoch", "Loss", "Accuracy", "Val loss", "Val accuracy"]] + [[h["epoch"], h["loss"], h["accuracy"], h["val_loss"], h["val_accuracy"]] for h in meta["history"]]
        s.append(Spacer(1, 6))
        s.append(table(rows))

    s.append(Paragraph("6. Model card", H2))
    s.append(table(MODEL_CARD, widths=[35 * mm, 143 * mm], header=False))
    s.append(Paragraph("7. Synthetic-data limits", H2))
    for lim in SYNTHETIC_LIMITS:
        s.append(Paragraph(f"- {lim}", SMALL))
    s.append(Paragraph("8. Licence & IP note", H2))
    s.append(table([["Component", "Licence", "Note"]] + LICENCES, widths=[55 * mm, 40 * mm, 83 * mm]))
    doc.build(s)
    return buf.getvalue()
