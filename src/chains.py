"""
chains.py - Persona prompting + tool-enabled LLM.

Plain message lists, no LCEL magic: build the system prompt, retrieve context
for evidence + grounding, and hand the model a tool so IT decides when to
consult another persona.
"""
from src.persona_loader import load_persona, build_system_prompt
from src.providers import get_llm, MODEL_ID
from src.retriever import search_with_scores
from src.evidence import build_evidence

# Last retrieval evidence per persona, read by chat_engine for the
# Retrieved Context panel.
_LAST_EVIDENCE = {}


def get_last_evidence(persona_id: str):
    return _LAST_EVIDENCE.get(persona_id)


_SYSTEM_CACHE = {}


def get_persona_system(persona_id: str) -> str:
    """System prompt for a persona, including how to use its tool."""
    if persona_id not in _SYSTEM_CACHE:
        persona = load_persona(persona_id)
        system = build_system_prompt(persona)
        # Keep persona voice but stay grounded: at most one metaphor/simile per
        # answer, no stacked poetic imagery, prioritize clear explanation.
        system += "\n\nSTYLE NOTE: Be direct and sparing with poetry — max one metaphor per answer."
        system += (
            "\n\nYou have a tool called consult_persona. When the user asks what "
            "another character would say or think, call it with that character and "
            "the question. The tool's reply is shown to the user directly as that "
            "character's answer, so add no commentary of your own — just call the tool. "
            "Otherwise just answer as yourself."
        )
        _SYSTEM_CACHE[persona_id] = system
    return _SYSTEM_CACHE[persona_id]


def retrieve_context(persona_id: str, question: str, k: int = 4) -> str:
    """Retrieve grounding context: logs hits, stashes panel evidence, returns text."""
    from src.telemetry import get_logger, timer
    log = get_logger("personachat.chains")
    with timer(log, "retrieve", persona=persona_id, model=MODEL_ID, qlen=len(question)):
        scored, elapsed_ms = search_with_scores(persona_id, question, k=k)
        docs = [d for d, _ in scored]
    log.info(f"retrieve hits={len(docs)} persona={persona_id} " +
             " ".join(f"{d.metadata.get('group', '?')}" for d in docs[:4]))
    # Show what RAG actually fetched (terminal + log file) so accuracy is visible.
    for i, d in enumerate(docs):
        preview = d.page_content[:400].replace("\n", " ")
        log.info(f"rag[{i}] group={d.metadata.get('group', '?')} len={len(d.page_content)} :: {preview}")
    # Stash UI-safe evidence (truncated cards + scores) for the Retrieved Context panel.
    persona = load_persona(persona_id)
    _LAST_EVIDENCE[persona_id] = build_evidence(
        scored, elapsed_ms, persona.get("display_name", persona_id)
    )
    return "\n\n".join(d.page_content for d in docs)


_LLM_CACHE = None


def get_persona_llm():
    """Single LLM instance with the consult_persona tool attached."""
    global _LLM_CACHE
    if _LLM_CACHE is None:
        from src.persona_tools import consult_persona
        _LLM_CACHE = get_llm(temperature=0.7).bind_tools([consult_persona])
    return _LLM_CACHE
