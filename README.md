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
  chains.py             # LCEL chain per persona (system + retrieved context)
  router.py             # detects "what would tesla say" -> invoked persona
  chat_engine.py        # orchestrates active vs temporary invoked, auto-switch back
chroma_db/              # persists embeddings (gitignored)
app.py                  # Streamlit UI
```

## Perfect behaviour adaptation
- `never_list` + `reaction_rules` are hard guardrails in `system_prompt` and also separate retrievable docs.
- Speaking style / thinking patterns retrieved via RAG each turn (k=4).

## Switching logic
Chatting with `albert_einstein`, user: `what would tesla say on wireless?`
1. `router.detect_invocation` -> `{invoked: nikola_tesla}`
2. `chat_engine` temporarily invokes `persona_nikola_tesla` retriever/chain
3. Response prefixed `[As Nikola Tesla]: ... [Back to Albert Einstein]`
4. Next turn auto back to `active_persona`.

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
2. `Chroma` collections + `OpenAIEmbeddings`
3. `as_retriever` + LCEL (`RunnableLambda | retriever | prompt | llm | StrOutputParser`)
4. `ChatPromptTemplate` + `MessagesPlaceholder` (history)
5. Router as classifier chain
