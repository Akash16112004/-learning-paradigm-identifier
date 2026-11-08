---
title: P2 Paradigm Classifier
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# P2 · DistilBERT Learning-Paradigm Classifier (Hugging Face Space)

This Space is the live DistilBERT part of the P2 project. It trains `distilbert-base-uncased` in Keras 3 (PyTorch backend) on the 200 LLM-generated scenarios in `synthetic_200.csv`, then classifies new scenarios and scores the 20 hand-labelled benchmark.

## How to publish
1. Create a free account at huggingface.co → New Space → SDK **Gradio**, hardware **CPU basic (free)**.
2. Upload every file in this folder (`app.py`, `trainer.py`, `data.py`, `guardrail.py`, `synthetic_200.csv`, `requirements.txt`, `README.md`).
3. Open the Space → **Train** tab → Start fine-tuning (about 5–10 minutes on the free CPU) → use **Classify** and **Benchmark**.
4. In the main app, add the Space name (e.g. `your-username/p2-paradigm-classifier`) as the `HF_SPACE_URL` secret. The Classify page then shows live DistilBERT answers from the Space, and the benchmark scores DistilBERT through it.

No API keys are needed here. The model weights are kept only while the Space is running, so you need to retrain after the Space restarts.
