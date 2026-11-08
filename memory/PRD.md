# PRD — P2 Learning-Paradigm Identifier

## Problem statement
Generative-AI mini-project P2: label a problem scenario as supervised / unsupervised / reinforcement learning and suggest an algorithm. U1 CoT prompt + token-prob confidence; U2 LLM generates 200 synthetic scenarios, zero-shot vs few-shot; U3 fine-tune DistilBERT and compare with the LLM; U4 Keras training + pipeline illustration per answer; U5 per-domain accuracy (bias), synthetic-data limits. Deliverables: classifier app, accuracy table on 20 hand-labelled scenarios, model card, live URL.

## Defaults chosen (user skipped questions)
gpt-4.1-mini via Emergent key (logprobs), real Keras 3 (torch backend) DistilBERT training in the app, Nano Banana image per answer, deploy on Emergent.

## Architecture
- backend: server.py (routes), llm.py (zero/few/CoT, logprob confidence, synthetic generation, illustration), trainer.py (Keras DistilBERT fine-tune, attention, PCA), guardrail.py, data.py (20 gold benchmark, few-shot, algorithm lists, limits)
- frontend: Classify, Synthetic data, Keras training, Evaluation, Foundations, Responsible AI pages
- Mongo collections: synthetic, classifications, eval_runs. Model weights: backend/artifacts/

## Implemented (2026-06)
- All 6 pages working end to end; tests iteration_1: backend 16/16 passed, frontend minor issues fixed
- Latest results: zero/few/CoT LLM 100% on 20 gold; DistilBERT 75% gold, 95% synthetic val; ~120 s CPU training

## Backlog
- P1: Hugging Face Spaces / Gradio version for course hosting requirement
- P1: Persist job state (in-memory now; lost on restart)
- P2: Harder gold set (hybrid paradigms) to separate LLM methods; calibration chart
- 2026-06: Deploy fix — torch pin; ML libs optional (saved DistilBERT results in prod); hf_space/ Gradio Space for live DistilBERT
- 2026-06: PDF report (/api/report.pdf), 10 tricky benchmark items (30 total), HF Space link via HF_SPACE_URL + gradio_client
