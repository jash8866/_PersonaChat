"""
app.py - Streamlit PersonaChat with LangChain RAG.
Run:
  pip install -r requirements.txt
  cp .env.example .env  # add OPENAI_API_KEY
  python -m src.ingestion   # first time: embed normalized_profile -> chroma_db
  streamlit run app.py
"""
import streamlit as st
from src.persona_loader import list_personas, load_persona
from src.chat_engine import ChatEngine
from src.providers import list_models, get_default_model_id

st.set_page_config(page_title="PersonaChat", layout="wide")
st.title("PersonaChat — RAG Persona Switching")
st.caption("Perfect personality adaptation via per-persona Chroma + LangChain. Try: chatting as Einstein, then ask 'what would Tesla say about wireless energy?'")

# Sidebar: persona + model selectors
personas = list_personas()
ids = [p["id"] for p in personas]
labels = {p["id"]: f"{p['display_name']} ({p['era']})" for p in personas}
models = list_models()
model_ids = [m["id"] for m in models]
model_labels = {m["id"]: m["label"] for m in models}

if "active" not in st.session_state:
    st.session_state.active = ids[0]
if "model" not in st.session_state:
    st.session_state.model = get_default_model_id()
if "engine" not in st.session_state:
    st.session_state.engine = ChatEngine(st.session_state.active, model_id=st.session_state.model)

with st.sidebar:
    st.header("Model")
    mchoice = st.selectbox("Choose model", model_ids, format_func=lambda x: model_labels[x], index=model_ids.index(st.session_state.model) if st.session_state.model in model_ids else 0)
    if mchoice != st.session_state.model:
        st.session_state.model = mchoice
        st.session_state.engine.set_model(mchoice)
        st.rerun()
    st.caption(f"Serving: {model_labels[st.session_state.model]}")
    st.divider()
    st.header("Active Persona")
    choice = st.selectbox("Choose who you talk to", ids, format_func=lambda x: labels[x], index=ids.index(st.session_state.active))
    if choice != st.session_state.active:
        st.session_state.active = choice
        st.session_state.engine.set_active(choice)
        st.rerun()
    st.divider()
    p = load_persona(st.session_state.active)
    st.subheader(p["display_name"])
    st.write(p["identity"].get("archetype",""))
    st.write(f"**Voice:** {p['speaking_style']['tone']}")
    if st.button("Clear history"):
        st.session_state.engine.clear()
        st.rerun()
    st.divider()
    st.markdown("**Invocation examples:**\n- what would tesla say on this?\n- how would newton respond?\n- respond as shelby")

# Render history
for msg in st.session_state.engine.get_history():
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input(f"Chat with {load_persona(st.session_state.active)['display_name']}"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking in character..."):
            try:
                resp, meta = st.session_state.engine.chat(prompt)
            except Exception as e:
                from src.telemetry import get_logger
                get_logger("personachat.app").exception(f"chat failed model={st.session_state.model}")
                st.error(f"Request failed ({type(e).__name__}): {e}. Check logs/personachat.log. Try a faster model or retry.")
                st.stop()
        st.markdown(resp)
        lat = f"{meta.get('elapsed', 0):.1f}s" if meta.get("elapsed") else "?"
        if meta.get("switch"):
            st.caption(f"Invoked: {meta['invoked']} -> back to {meta['active']} | {meta.get('model')} | {lat}")
        else:
            st.caption(f"{meta.get('model')} | {lat}")
