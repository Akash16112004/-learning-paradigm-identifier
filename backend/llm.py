import os
import re
import json
import math
import time
import uuid
import asyncio
from emergentintegrations.llm.chat import LlmChat, UserMessage
from data import LETTERS, FEW_SHOT, KNOWN_ALGOS, DOMAINS

MODEL = "gpt-4.1-mini"
IMAGE_MODEL = "gemini-3.1-flash-image-preview"

LETTER_SYSTEM = (
    "You classify machine-learning problem scenarios by learning paradigm.\n"
    "A = supervised learning (learn a mapping from labelled examples)\n"
    "B = unsupervised learning (find structure in unlabelled data)\n"
    "C = reinforcement learning (agent learns actions from rewards through interaction)\n"
    "D = not a machine-learning problem / off-topic\n"
    "Reply with exactly one capital letter: A, B, C or D."
)

COT_SYSTEM = (
    "You are an ML tutor. Think step by step to decide the learning paradigm of a problem scenario.\n"
    "Step 1: identify the data and whether target labels exist. Step 2: check whether an agent acts and receives rewards. "
    "Step 3: decide the paradigm. Step 4: suggest one concrete algorithm and two alternatives.\n"
    "Paradigm letters: A=supervised, B=unsupervised, C=reinforcement, D=not an ML problem.\n"
    "Return ONLY JSON: {\"steps\": [\"...\", \"...\", \"...\"], \"answer\": \"A|B|C|D\", \"algorithm\": \"...\", "
    "\"alternatives\": [\"...\", \"...\"], \"pipeline\": [\"5-6 short pipeline stage names\"]}"
)


def _key():
    return os.environ["EMERGENT_LLM_KEY"]


def _chat(system, model=MODEL, provider="openai", **params):
    return LlmChat(api_key=_key(), session_id=str(uuid.uuid4()), system_message=system).with_model(provider, model).with_params(**params)


def _parse_json(text):
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    return json.loads(m.group(1) if m else text)


async def letter_probs(user_text, system=LETTER_SYSTEM):
    """One-token answer; confidence = softmax over top token log-probabilities."""
    t0 = time.perf_counter()
    chat = _chat(system, logprobs=True, top_logprobs=10, max_tokens=1, temperature=0)
    r = await chat.send_message_with_tools(UserMessage(text=user_text))
    latency = (time.perf_counter() - t0) * 1000
    probs = {k: 0.0 for k in LETTERS}
    raw_tokens = []
    lp = r.raw.choices[0].logprobs
    if lp and lp.content:
        for t in lp.content[0].top_logprobs:
            tok, val = getattr(t, "token", None), getattr(t, "logprob", None)
            raw_tokens.append({"token": tok, "logprob": round(val, 4), "prob": round(math.exp(val), 6)})
            k = (tok or "").strip().upper()
            if k in probs:
                probs[k] += math.exp(val)
    total = sum(probs.values())
    if total == 0:
        k = (r.content or "D").strip().upper()[:1]
        probs[k if k in probs else "D"] = 1.0
        total = 1.0
    named = {LETTERS[k]: round(v / total, 4) for k, v in probs.items()}
    label = max(named, key=named.get)
    return {"label": label, "probs": named, "confidence": named[label], "top_tokens": raw_tokens, "latency_ms": round(latency)}


def few_shot_prompt(text):
    shots = "\n\n".join(f"Scenario: {s}\nAnswer: {a}" for s, a in FEW_SHOT)
    return f"{shots}\n\nScenario: {text}\nAnswer:"


async def zero_shot(text):
    return await letter_probs(f"Scenario: {text}\nAnswer:")


async def few_shot(text):
    return await letter_probs(few_shot_prompt(text))


def algorithm_grounded(paradigm, algo):
    a = (algo or "").lower()
    return any(k in a for k in KNOWN_ALGOS.get(paradigm, []))


async def chain_of_thought(text):
    t0 = time.perf_counter()
    chat = _chat(COT_SYSTEM, temperature=0)
    raw = await chat.send_message(UserMessage(text=f"Scenario: {text}"))
    try:
        cot = _parse_json(raw)
    except Exception:
        cot = {"steps": [raw], "answer": "D", "algorithm": "", "alternatives": [], "pipeline": []}
    reasoning = " ".join(cot.get("steps", []))
    final = await letter_probs(f"Scenario: {text}\nReasoning: {reasoning}\nFinal answer letter:")
    final["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    paradigm = final["label"]
    return {
        **final,
        "steps": cot.get("steps", []),
        "stated_answer": LETTERS.get(str(cot.get("answer", "D")).strip().upper()[:1], "not_ml"),
        "algorithm": cot.get("algorithm", ""),
        "alternatives": cot.get("alternatives", []),
        "pipeline": cot.get("pipeline", []),
        "algorithm_grounded": algorithm_grounded(paradigm, cot.get("algorithm", "")),
    }


async def generate_domain(domain, n=25):
    system = "You write realistic, varied machine-learning problem scenarios for a dataset. Return only JSON."
    prompt = (
        f"Write {n} distinct problem scenarios from the {domain} domain. Exactly {n - 2 * (n // 3)} must be supervised, "
        f"{n // 3} unsupervised and {n // 3} reinforcement learning problems. Each is 1-2 sentences describing the data and goal "
        "as a business or research need. Do NOT name the paradigm or algorithm in the text. Vary length and wording; include some "
        "harder cases. Return a JSON array of objects: {\"text\": str, \"label\": \"supervised|unsupervised|reinforcement\", "
        "\"algorithm\": str}."
    )
    chat = _chat(system, temperature=0.9)
    raw = await chat.send_message(UserMessage(text=prompt))
    items = _parse_json(raw)
    out = []
    for it in items:
        lab = str(it.get("label", "")).lower().strip()
        if lab in ("supervised", "unsupervised", "reinforcement") and it.get("text"):
            out.append({"text": it["text"].strip(), "label": lab, "algorithm": it.get("algorithm", ""), "domain": domain})
    return out[:n]


async def generate_all(per_domain=25):
    results = await asyncio.gather(*[generate_domain(d, per_domain) for d in DOMAINS], return_exceptions=True)
    rows = []
    for r in results:
        if isinstance(r, list):
            rows.extend(r)
    return rows


async def illustrate(scenario, paradigm, algorithm, pipeline):
    stages = " -> ".join(pipeline) if pipeline else "data -> preprocessing -> model -> evaluation -> deployment"
    prompt = (
        f"A clean, flat-vector technical infographic of a machine learning pipeline, left-to-right flow with arrows and icons. "
        f"Stages: {stages}. Paradigm: {paradigm} learning, model: {algorithm}. Context: {scenario[:220]}. "
        "Dark navy background, cyan, amber and emerald accents, minimal readable labels, no people, no logos."
    )
    chat = _chat("You create technical illustrations.", model=IMAGE_MODEL, provider="gemini").with_params(modalities=["image", "text"])
    _, images = await chat.send_message_multimodal_response(UserMessage(text=prompt))
    if not images:
        return None
    return {"mime_type": images[0]["mime_type"], "data": images[0]["data"], "prompt": prompt}
