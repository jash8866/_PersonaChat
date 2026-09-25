"""
chat_engine.py - Orchestrates active persona + temporary invoked persona.
Perfect switching: Einstein chat -> "what would Tesla say" -> Tesla answers -> next turn auto back to Einstein.
State is explicit: no hidden mutation.
"""
from typing import List, Dict, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from src.chains import get_chain
from src.router import detect_invocation
from src.persona_loader import load_persona

class ChatEngine:
    def __init__(self, active_persona: str, model_id: str | None = None):
        from src.providers import get_default_model_id
        self.active_persona = active_persona
        self.model_id = model_id or get_default_model_id()
        self.history: List[HumanMessage | AIMessage] = []

    def set_active(self, persona_id: str):
        self.active_persona = persona_id

    def set_model(self, model_id: str):
        self.model_id = model_id

    def _invoke(self, persona_id: str, query: str) -> str:
        from src.telemetry import get_logger, timer
        log = get_logger("personachat.engine")
        with timer(log, "llm_invoke", persona=persona_id, model=self.model_id, qlen=len(query)):
            chain = get_chain(persona_id, model_id=self.model_id)
            return chain.invoke({"question": query, "history": self.history})

    def chat(self, user_input: str) -> Tuple[str, Dict]:
        """
        Returns (response_text, meta)
        meta = {"invoked": str|None, "active": str, "model": str, "elapsed": float}
        """
        import time
        from src.telemetry import get_logger
        log = get_logger("personachat.engine")
        start = time.perf_counter()
        log.info(f"chat start active={self.active_persona} model={self.model_id} qlen={len(user_input)} q={user_input[:120]!r}")
        route = detect_invocation(user_input, self.active_persona)
        invoked = route["invoked"]
        log.info(f"route invoked={invoked} active={self.active_persona}")

        if invoked and invoked != self.active_persona:
            # Temporary persona answers (same model, different persona chain)
            tesla_answer = self._invoke(invoked, route["query"] or user_input)
            # Record: user -> tesla answer -> but history keeps both personas tagged
            self.history.append(HumanMessage(content=user_input))
            # Prefix makes switch explicit in transcript
            tagged = f"[As {load_persona(invoked)['display_name']}]: {tesla_answer}\n\n[Back to {load_persona(self.active_persona)['display_name']}]"
            self.history.append(AIMessage(content=tagged))
            elapsed = time.perf_counter() - start
            log.info(f"chat done switch={invoked} elapsed={elapsed:.2f}s")
            return tagged, {"invoked": invoked, "active": self.active_persona, "switch": True, "model": self.model_id, "elapsed": elapsed}

        # Normal active persona answer
        answer = self._invoke(self.active_persona, user_input)
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=answer))
        elapsed = time.perf_counter() - start
        log.info(f"chat done switch=None elapsed={elapsed:.2f}s alen={len(answer)}")
        return answer, {"invoked": None, "active": self.active_persona, "switch": False, "model": self.model_id, "elapsed": elapsed}

    def get_history(self) -> List[Dict]:
        out = []
        for m in self.history:
            out.append({"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content})
        return out

    def clear(self):
        self.history = []
