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

_SECTION_ICONS = {
    "Identity": "👤",
    "Rule": "⚙️",
    "Rules": "⚙️",
    "Voice": "💬",
}

def _render_evidence(evidence):
    """Compact Retrieved Context panel below an assistant response.

    UI organization only: renders the already-retrieved chunks stored in
    chat metadata. Shows truncated previews prepared server-side in
    src/evidence.py — full chunk text never reaches the UI. Unrecognized
    types fall back to "Context"; relevance scores are displayed as provided.
    """
    if not evidence or not evidence.get("chunks"):
        return
    count = evidence.get("count", len(evidence["chunks"]))
    ms = evidence.get("retrieval_ms", 0)
    with st.expander(f"🗂️ Retrieved context · {count} items", expanded=False):
        st.markdown("**Retrieved Context**")
        st.caption(f"{count} retrieved items · {ms} ms")
        for chunk in evidence["chunks"]:
            meta = chunk.get("meta", {})
            section = meta.get("section") or "Context"
            icon = _SECTION_ICONS.get(section, "📄")
            pct_val = min(100, max(0, int(chunk.get("relevance", 0))))
            with st.container(border=True):
                head, pct = st.columns([5, 1])
                head.markdown(f"{icon} **{section}**")
                pct.markdown(f"**{pct_val}%**")
                st.progress(pct_val)
                st.markdown(chunk.get("title", "Source"))
                st.caption(chunk.get("source_type", ""))
                st.markdown(f"> {chunk.get('preview', '')}")
                with st.popover("View Source"):
                    st.markdown("**Retrieved passage**")
                    st.markdown(f"> {chunk.get('excerpt', '')}")
                    meta_bits = [f"Section: {section}"]
                    if meta.get("cited"):
                        meta_bits.append(f"Cited: {meta['cited']}")
                    st.caption(" · ".join(meta_bits))

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
if "evidence" not in st.session_state:
    # Parallel to engine history: one entry per message (None for user msgs
    # and answers without retrieval evidence).
    st.session_state.evidence = []

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
        st.session_state.evidence = []
        st.rerun()
    st.divider()
    st.markdown("**Invocation examples:**\n- what would tesla say on this?\n- how would newton respond?\n- respond as shelby")

# Render history
history = st.session_state.engine.get_history()
for i, msg in enumerate(history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.evidence):
            _render_evidence(st.session_state.evidence[i])

# Input
if prompt := st.chat_input(f"Chat with {load_persona(st.session_state.active)['display_name']}"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking in character..."):
            try:
                resp, meta = st.session_state.engine.chat(prompt)
                # History grew by 2 (user + assistant); keep evidence aligned.
                st.session_state.evidence.extend([None, meta.get("evidence")])
            except Exception as e:
                from src.telemetry import get_logger
                get_logger("personachat.app").exception(f"chat failed model={st.session_state.model}")
                st.error(f"Request failed ({type(e).__name__}): {e}. Check logs/personachat.log. Try a faster model or retry.")
                st.stop()
        st.markdown(resp)
        _render_evidence(meta.get("evidence"))
        lat = f"{meta.get('elapsed', 0):.1f}s" if meta.get("elapsed") else "?"
        if meta.get("switch"):
            st.caption(f"Invoked: {meta['invoked']} -> back to {meta['active']} | {meta.get('model')} | {lat}")
        else:
            st.caption(f"{meta.get('model')} | {lat}")
