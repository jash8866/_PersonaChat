import json
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.persona_loader import list_personas, load_persona
from src.providers import get_embeddings, EMBED_MODEL

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))

CHUNK_GROUPS = [
    "core_traits",
    "speaking_style",
    "thinking_patterns",
    "never_list",
    "reaction_rules",
    "backstory",
    "episodes",
    "relationships",
    "preferences",
    "canonical_quotes",
    "micro_behaviors",
    "vocabulary",
]


def persona_to_documents(persona: dict) -> list[Document]:
    pid = persona["id"]
    docs: list[Document] = []

    docs.append(Document(
        page_content=f"IDENTITY: {persona['display_name']} - {persona['system_prompt']}",
        metadata={"persona": pid, "group": "system_prompt", "weight": "high"}
    ))

    for group in CHUNK_GROUPS:
        value = persona.get(group)
        if not value:
            continue
        content = f"[{group.upper()} - {pid}]\n{json.dumps(value, ensure_ascii=False, indent=2)}"
        docs.append(Document(
            page_content=content,
            metadata={"persona": pid, "group": group}
        ))

    for rule in persona.get("reaction_rules", []):
        docs.append(Document(
            page_content=f"REACTION RULE [{pid}]: IF {rule['if']} THEN {rule['then']}",
            metadata={"persona": pid, "group": "reaction_rule_single"}
        ))
    for n in persona.get("never_list", []):
        docs.append(Document(
            page_content=f"NEVER [{pid}]: {n['prohibition']} | Evidence: {n.get('evidence','')}",
            metadata={"persona": pid, "group": "never_single"}
        ))
    for q in persona.get("canonical_quotes", []):
        docs.append(Document(
            page_content=f"QUOTE [{pid}]: {q}",
            metadata={"persona": pid, "group": "quote"}
        ))
    return docs


def ingest_all(reset: bool = True):
    embeddings = get_embeddings()
    personas = list_personas()
    print(f"Ingesting {len(personas)} personas -> {CHROMA_DIR} (model={EMBED_MODEL})")
    for p in personas:
        pid = p["id"]
        data = load_persona(pid)
        docs = persona_to_documents(data)
        collection_name = f"persona_{pid}"
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        if reset:
            try:
                store.delete_collection()
                store = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=str(CHROMA_DIR),
                )
            except Exception:
                pass
        store.add_documents(docs)
        print(f"  + {pid}: {len(docs)} docs -> collection {collection_name}")

    print("Done. Verify with: Chroma(persist_directory, collection_name='persona_albert_einstein').get()")


if __name__ == "__main__":
    ingest_all()
