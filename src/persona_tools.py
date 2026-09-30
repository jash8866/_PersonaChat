"""
persona_tools.py - Tools the LLM can call while answering as a persona.

Instead of regexes/aliases/a router deciding when to switch personas, the
model itself gets one tool: consult_persona. When the user asks what another
character would say, the model calls it and works the reply into its answer.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.persona_loader import list_personas, load_persona, build_system_prompt
from src.providers import get_llm
from src.retriever import search_with_scores


def _valid_ids():
    return [p["id"] for p in list_personas()]


@tool
def consult_persona(persona_id: str, question: str) -> str:
    """Ask another character for their perspective on a question.

    Use this when the user wants to hear what a different character would
    say or think. persona_id must be a valid character id, question is what
    to ask them. Returns that character's in-character reply.
    """
    ids = _valid_ids()
    if persona_id not in ids:
        return f"Unknown persona '{persona_id}'. Available: {', '.join(ids)}."
    persona = load_persona(persona_id)
    system = build_system_prompt(persona)
    system += "\n\nSTYLE NOTE: Be direct and sparing with poetry — max one metaphor per answer."
    system += ("\n\nYour reply will be shown to the user directly as your words, "
               "so answer the question fully in your own voice.")
    scored, _ = search_with_scores(persona_id, question, k=4)
    context = "\n\n".join(d.page_content for d, _ in scored)
    llm = get_llm(temperature=0.7)
    resp = llm.invoke([
        SystemMessage(content=system + "\n\nRETRIEVED CONTEXT (use to stay in character):\n" + context),
        HumanMessage(content=question),
    ])
    return resp.content or "(no reply)"


# List the real ids in the tool description so the model knows who it can ask.
consult_persona.description = (
    "Ask another character for their perspective on a question. "
    "Use this when the user wants to hear what a different character would say or think. "
    f"persona_id must be one of: {', '.join(_valid_ids())}. "
    "question is what to ask them. Returns that character's in-character reply."
)
