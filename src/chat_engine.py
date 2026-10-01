from typing import List, Dict, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.chains import (
    get_persona_system,
    retrieve_context,
    get_persona_llm,
    get_last_evidence,
)
from src.persona_tools import consult_persona
from src.persona_loader import load_persona
from src.providers import MODEL_ID


class ChatEngine:
    def __init__(self, active_persona: str):
        self.active_persona = active_persona
        self.history: List[HumanMessage | AIMessage] = []

    def set_active(self, persona_id: str):
        self.active_persona = persona_id

    def _invoke(self, persona_id: str, query: str) -> Tuple[str, List[str]]:
        from src.telemetry import get_logger, timer
        log = get_logger("personachat.engine")
        with timer(log, "llm_invoke", persona=persona_id, model=MODEL_ID, qlen=len(query)):
            system = get_persona_system(persona_id)
            context = retrieve_context(persona_id, query)
            llm = get_persona_llm()
            reply = llm.invoke([
                SystemMessage(content=system + "\n\nRETRIEVED CONTEXT (use to stay in character):\n" + context),
                *self.history,
                HumanMessage(content=query),
            ])
            consulted: List[str] = []
            outputs: List[Tuple[str, str]] = []
            for call in getattr(reply, "tool_calls", None) or []:
                if call["name"] != "consult_persona":
                    continue
                pid = (call.get("args") or {}).get("persona_id", "")
                log.info(f"tool call consult_persona persona={pid}")
                result = str(consult_persona.invoke(call.get("args") or {}))
                if pid and pid not in consulted:
                    consulted.append(pid)
                outputs.append((pid, result))
            if outputs:
                if len(outputs) == 1:
                    return outputs[0][1], consulted
                parts = []
                for pid, text in outputs:
                    try:
                        name = load_persona(pid)["display_name"]
                    except Exception:
                        name = pid or "Unknown"
                    parts.append(f"[{name}]:\n{text}")
                return "\n\n".join(parts), consulted
            return (reply.content or "(no answer)"), consulted

    def chat(self, user_input: str) -> Tuple[str, Dict]:
        import time
        from src.telemetry import get_logger
        log = get_logger("personachat.engine")
        start = time.perf_counter()
        log.info(f"chat start active={self.active_persona} model={MODEL_ID} qlen={len(user_input)} q={user_input[:120]!r}")

        answer, consulted = self._invoke(self.active_persona, user_input)
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=answer))
        elapsed = time.perf_counter() - start
        first = consulted[0] if consulted else None
        log.info(f"chat done consulted={consulted} elapsed={elapsed:.2f}s alen={len(answer)}")
        return answer, {
            "invoked": first,
            "active": self.active_persona,
            "switch": bool(consulted),
            "consulted": consulted,
            "model": MODEL_ID,
            "elapsed": elapsed,
            "evidence": get_last_evidence(self.active_persona),
        }

    def get_history(self) -> List[Dict]:
        out = []
        for m in self.history:
            out.append({"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content})
        return out

    def clear(self):
        self.history = []
