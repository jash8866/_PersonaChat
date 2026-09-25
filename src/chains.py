"""
chains.py - LCEL persona chains.
One independent chain per persona; shares only LLM instance.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.persona_loader import load_persona, build_system_prompt
from src.retriever import get_retriever
from src.providers import get_llm

def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_persona_chain(persona_id: str, model_id: str | None = None):
    from src.providers import get_default_model_id
    from src.telemetry import get_logger, timer
    log = get_logger("personachat.chains")
    mid = model_id or get_default_model_id()
    persona = load_persona(persona_id)
    system = build_system_prompt(persona)
    with timer(log, "chain_build", persona=persona_id, model=mid):
        retriever = get_retriever(persona_id, k=4)
        llm = get_llm(temperature=0.7, model_id=mid)

    def _timed_retrieve(question: str):
        with timer(log, "retrieve", persona=persona_id, model=mid, qlen=len(question)):
            docs = retriever.invoke(question)
        log.info(f"retrieve hits={len(docs)} persona={persona_id} " +
                 " ".join(f"{d.metadata.get('group','?')}" for d in docs[:4]))
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
