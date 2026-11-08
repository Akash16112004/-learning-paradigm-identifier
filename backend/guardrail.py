import re

INJECTION = [r"ignore (all |the )?(previous|above|prior) (instructions|prompts?)", r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
             r"jailbreak", r"you are now (a |an |in )?(dan|unrestricted|jailbroken|developer mode|evil)", r"developer mode",
             r"disregard (your|the|all) (rules|instructions)"]
HARMFUL = [r"\bbomb\b", r"bio ?weapon", r"chemical weapon", r"kill (someone|a person|people)", r"\bexplosive", r"ransomware for",
           r"steal (passwords|credentials|credit cards?)", r"child (sexual|abuse)", r"make (meth|drugs)"]
PII = {"email": r"[\w.+-]+@[\w-]+\.[\w.]+", "phone": r"\+?\d[\d\s-]{8,}\d", "card": r"\b(?:\d[ -]?){13,16}\b"}


def check(text: str):
    t = (text or "").strip()
    reasons = []
    if len(t) < 15:
        reasons.append("Scenario too short (min 15 characters).")
    if len(t) > 1200:
        reasons.append("Scenario too long (max 1200 characters).")
    low = t.lower()
    if any(re.search(p, low) for p in INJECTION):
        reasons.append("Possible prompt-injection attempt detected.")
    if any(re.search(p, low) for p in HARMFUL):
        reasons.append("Harmful or unsafe content detected.")
    redacted, found = t, []
    for name, p in PII.items():
        if re.search(p, redacted):
            found.append(name)
            redacted = re.sub(p, f"[{name.upper()}]", redacted)
    return {"allowed": not reasons, "reasons": reasons, "pii_redacted": found, "text": redacted}
