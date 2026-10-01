# PersonaChat — RAG Persona Switching (LangChain)

Modular, independent persona system: one JSON per personality, one Chroma collection per persona.

## Structure
```
normalized_profile/
  _schema.json          # validation contract
  index.json            # registry (optional)
  albert_einstein.json  # self-contained
  isaac_newton.json
  nikola_tesla.json
  odysseus.json
  pablo_escobar.json
  thomas_shelby.json
src/
  persona_loader.py     # independent loader, builds system_prompt
  ingestion.py          # chunk by keys -> Chroma per-persona
  retriever.py          # get_retriever(persona_id)
  chains.py             # ChatPromptTemplate + LCEL chain per persona (system + retrieved context)
  persona_tools.py      # consult_persona tool: one persona asks another
  chat_engine.py        # chat loop; model calls the tool itself when asked
chroma_db/              # persists embeddings (gitignored)
app.py                  # Streamlit UI
```

## Perfect behaviour adaptation
- `never_list` + `reaction_rules` are hard guardrails in `system_prompt` and also separate retrievable docs.
- Speaking style / thinking patterns retrieved via RAG each turn (k=4).

## Switching logic (tool calling, no router)
Chatting with `albert_einstein`, user: `what would tesla say on wireless?`
1. Einstein's LLM sees the `consult_persona` tool and calls it with `(nikola_tesla, "...wireless?")`
2. The tool answers as Tesla and that reply IS the answer — shown directly, no commentary from Einstein
3. No keyword lists, no aliases

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # set OPENAI_API_KEY
python -m src.ingestion  # first time only
streamlit run app.py
```
Set `LLM_MODEL=gpt-4o-mini` and `EMBEDDING_MODEL=text-embedding-3-small` in `.env`.
Local Ollama alternative: change `ChatOpenAI` to `ChatOllama` in `src/chains.py`.

## LangChain learning path in this repo
1. `Document` + per-group chunking (`src/ingestion.py:persona_to_documents`)
2. `Chroma` collections + `HuggingFaceEmbeddings` (local)
3. `ChatPromptTemplate` + `MessagesPlaceholder` (history) piped to the tool-enabled LLM via LCEL (`prompt | llm`)
4. `@tool` + `llm.bind_tools` with direct tool-output answers (`src/chat_engine.py`)
