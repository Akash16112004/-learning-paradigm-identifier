# P2 · Learning-Paradigm Identifier

Classifies a problem scenario as **supervised, unsupervised or reinforcement learning** and suggests an algorithm.

| Unit | What is implemented |
|---|---|
| U1 | Chain-of-thought prompt; confidence from token log-probabilities; zero-shot vs few-shot vs CoT; lifecycle diagram; generative vs discriminative; PCA latent-space view |
| U2 | LLM generates 200 synthetic scenarios; prompt-based learning vs fine-tuning; attention explanation + real DistilBERT attention |
| U3 | GPT-style generator (gpt-4.1-mini) + BERT-family classifier (DistilBERT); image modality (pipeline illustration) |
| U4 | Python, Keras 3 on PyTorch for training (`backend/trainer.py`); Nano Banana image generated per answer |
| U5 | Per-domain accuracy (bias), algorithm-grounding test (hallucination), input guardrail, licence note, `MODEL_CARD.md` |

## Stack
FastAPI + MongoDB backend (`backend/`), React dashboard (`frontend/`). LLM: gpt-4.1-mini; Image: gemini-3.1-flash-image-preview; Classifier: distilbert-base-uncased.

## Secrets
`EMERGENT_LLM_KEY`, `MONGO_URL`, `DB_NAME` come from environment / platform secrets — never commit them.

## Use
1. Synthetic data → Generate 200. 2. Keras training → Start fine-tuning (~2 min CPU). 3. Evaluation → Run benchmark. 4. Classify any scenario.

## Deployment note
The Emergent production host is too small for PyTorch, so torch/keras/transformers are not in `backend/requirements.txt`. In production the app runs GPT live and shows the saved DistilBERT results (`backend/artifacts/*.json`). The live DistilBERT model (training + inference) runs as a Hugging Face Space from `hf_space/` (see `hf_space/README.md`).
