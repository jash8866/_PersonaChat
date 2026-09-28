"""
persona_loader.py - Modular, independent persona loading.
Each normalized_profile/{id}.json is self-contained; no cross-file dependency.
Validated against _schema.json.
"""
import json
from pathlib import Path
from typing import Dict, List

BASE = Path(__file__).resolve().parent.parent
PROFILE_DIR = BASE / "normalized_profile"
INDEX_FILE = PROFILE_DIR / "index.json"
SCHEMA_FILE = PROFILE_DIR / "_schema.json"

def list_personas() -> List[Dict]:
    data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return data["personas"]

def load_persona(persona_id: str) -> Dict:
    """Load single persona - fully independent, no other files needed."""
    path = PROFILE_DIR / f"{persona_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Persona not found: {persona_id} -> {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    # light validation - required keys
    for key in ["id", "system_prompt", "never_list", "reaction_rules", "speaking_style"]:
        if key not in data:
            raise ValueError(f"Persona {persona_id} missing required key: {key}")
    return data

def build_system_prompt(persona: Dict) -> str:
    """Compose the strict system prompt - modular, no external context."""
    never = "\n".join(f"- NEVER {n['prohibition']}" for n in persona.get("never_list", []))
    rules = "\n".join(f"- IF {r['if']} THEN {r['then']}" for r in persona.get("reaction_rules", []))
    style = persona.get("speaking_style", {})
    return f"""
{persona['system_prompt']}

VOICE:
Tone: {style.get('tone','')}
Cadence: {style.get('cadence','')}
Diction: {style.get('diction','')}

HARD GUARDRAILS (must never violate):
{never}

REACTION RULES:
{rules}

Keep knowledge cutoff: {persona.get('identity', {}).get('knowledge_cutoff','')}
Vocabulary anchors: {', '.join(persona.get('vocabulary', [])[:8])}
"""

def get_persona_ids() -> List[str]:
    return [p["id"] for p in list_personas()]
