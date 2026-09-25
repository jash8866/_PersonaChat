"""
router.py - Detects temporary persona invocation.
E.g., chatting with einstein, user says "what would tesla say about X"
-> {invoked: tesla, query: X}
Returns None if no invocation.
"""
import re
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.persona_loader import get_persona_ids

# Fast regex pre-check (no LLM call if no signal)
PATTERNS = [
    r"what would (\w+) say",
    r"how would (\w+) (respond|react|answer|think)",
    r"as (\w+), what",
    r"respond as (\w+)",
    r"what does (\w+) think",
    r"\btesla\b.*say", r"\beinstein\b.*say", r"\bnewton\b.*say",
    r"\bodysseus\b", r"\bescobar\b", r"\bshelby\b", r"\bthomas\b"
]

ALIASES = {
    "einstein": "albert_einstein",
    "albert": "albert_einstein",
    "newton": "isaac_newton",
    "isaac": "isaac_newton",
    "tesla": "nikola_tesla",
    "nikola": "nikola_tesla",
    "odysseus": "odysseus",
    "ulysses": "odysseus",
    "escobar": "pablo_escobar",
    "pablo": "pablo_escobar",
    "shelby": "thomas_shelby",
    "thomas": "thomas_shelby",
    "tommy": "thomas_shelby",
}

def heuristic_detect(text: str) -> str | None:
    low = text.lower()
    for pat in PATTERNS:
        m = re.search(pat, low)
        if m:
            # try to extract persona token
            token = m.group(1) if m.lastindex else None
            if token and token.lower() in ALIASES:
                return ALIASES[token.lower()]
            # fallback: find any alias mentioned
            for alias, pid in ALIASES.items():
                if alias in low:
                    return pid
    # also direct alias mention with invocation intent words
    if any(w in low for w in ["what would", "how would", "as ", "respond as"]):
        for alias, pid in ALIASES.items():
            if alias in low:
                return pid
    return None

# LLM classifier for ambiguous cases - only called if heuristic uncertain
def _get_classifier_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages([
        ("system", """You are a router. Given user message and active persona, detect if user asks for a TEMPORARY response AS another persona.
Available personas: {personas}
Active persona: {active}
Return JSON: {{"invoked\": \"<id or null>\", \"query\": \"<the question to answer as invoked or null>\"}}
Only return invoked if user explicitly wants to hear what another persona would say. Not if they just mention a name in passing. Aliases: einstein->albert_einstein, tesla->nikola_tesla, newton->isaac_newton, odysseus->odysseus, escobar->pablo_escobar, shelby/tommy->thomas_shelby"""),
        ("human", "{text}")
    ])

def llm_detect(text: str, active_persona: str) -> dict:
    from src.providers import get_llm
    llm = get_llm(temperature=0)
    chain = _get_classifier_prompt() | llm
    resp = chain.invoke({
        "personas": ", ".join(get_persona_ids()),
        "active": active_persona,
        "text": text
    })
    import json, re
    content = resp.content
    # extract json
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(0))
            inv = j.get("invoked")
            if inv and inv not in get_persona_ids():
                # resolve alias
                inv = ALIASES.get(inv.lower(), inv)
            return {"invoked": inv if inv in get_persona_ids() else None, "query": j.get("query") or text}
        except Exception:
            pass
    return {"invoked": None, "query": None}

def detect_invocation(text: str, active_persona: str) -> dict:
    """Public API: returns {{invoked: str|None, query: str|None}}"""
    h = heuristic_detect(text)
    if h and h != active_persona:
        return {"invoked": h, "query": text}
    # if heuristic found same as active, no invocation
    if h == active_persona:
        return {"invoked": None, "query": None}
    # if heuristic found nothing but text looks like invocation, ask LLM
    if any(w in text.lower() for w in ["would", "as ", "say", "think"]):
        return llm_detect(text, active_persona)
    return {"invoked": None, "query": None}
