"""
chains.py - LCEL persona chains.
One independent chain per persona; shares only LLM instance.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.persona_loader import load_persona, build_system_prompt
from src.retriever import search_with_scores
from src.evidence import build_evidence
from src.providers import get_llm

def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Last retrieval evidence per (persona, model), read by chat_engine for the
# Evidence panel. Single-user Streamlit use is sequential, so a side channel
# avoids double retrieval (once for evidence, once for the answer).
_LAST_EVIDENCE = {}

def get_last_evidence(persona_id: str, model_id: str | None = None):
    from src.providers import get_default_model_id
    return _LAST_EVIDENCE.get((persona_id, model_id or get_default_model_id()))

def build_persona_chain(persona_id: str, model_id: str | None = None):
    from src.providers import get_default_model_id
    from src.telemetry import get_logger, timer
    log = get_logger("personachat.chains")
    mid = model_id or get_default_model_id()
    persona = load_persona(persona_id)
    system = build_system_prompt(persona)
    # Keep persona voice but stay grounded: at most one metaphor/simile per
    # answer, no stacked poetic imagery, prioritize clear explanation.
    system += "\n\nSTYLE NOTE: Be direct and sparing with poetry — max one metaphor per answer."
    with timer(log, "chain_build", persona=persona_id, model=mid):
        llm = get_llm(temperature=0.7, model_id=mid)

    def _timed_retrieve(question: str):
        with timer(log, "retrieve", persona=persona_id, model=mid, qlen=len(question)):
            scored, elapsed_ms = search_with_scores(persona_id, question, k=4)
            docs = [d for d, _ in scored]
        log.info(f"retrieve hits={len(docs)} persona={persona_id} " +
                 " ".join(f"{d.metadata.get('group','?')}" for d in docs[:4]))
        # Show what RAG actually fetched (terminal + log file) so accuracy is visible.
        for i, d in enumerate(docs):
            preview = d.page_content[:400].replace("\n", " ")
            log.info(f"rag[{i}] group={d.metadata.get('group','?')} len={len(d.page_content)} :: {preview}")
        # Stash UI-safe evidence (truncated cards + scores) for the Evidence panel.
        _LAST_EVIDENCE[(persona_id, mid)] = build_evidence(
            scored, elapsed_ms, persona.get("display_name", persona_id)
        )
        return docs

    prompt = ChatPromptTemplate.from_messages([
        ("system", system + "\n\nRETRIEVED CONTEXT (use to stay in character):\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | RunnableLambda(_timed_retrieve) | RunnableLambda(_format_docs),
            "question": RunnablePassthrough() | (lambda x: x["question"]),
            "history": RunnablePassthrough() | (lambda x: x.get("history", [])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Cache keyed by (persona, model) - switching model builds fresh chain, old ones reused
_CHAIN_CACHE = {}
def get_chain(persona_id: str, model_id: str | None = None):
    from src.providers import get_default_model_id
    key = (persona_id, model_id or get_default_model_id())
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = build_persona_chain(persona_id, model_id=key[1])
    return _CHAIN_CACHE[key]
