import json
import re

PREVIEW_MAX = 180
EXCERPT_MAX = 320

_HEADER_RE = re.compile(r"^\[[A-Z_ ]+ - [a-z_]+\]\s*")

GROUP_INFO = {
    "system_prompt": ("Identity", "Persona identity"),
    "episodes": ("Episode", "Profile narrative"),
    "backstory": ("Backstory", "Profile narrative"),
    "quote": ("Quote", "Profile narrative"),
    "canonical_quotes": ("Quotes", "Profile narrative"),
    "never_single": ("Rule", "Character rule"),
    "never_list": ("Rules", "Character rule"),
    "reaction_rule_single": ("Rule", "Behavior rule"),
    "reaction_rules": ("Rules", "Behavior rule"),
    "speaking_style": ("Voice", "Voice guide"),
    "thinking_patterns": ("Thinking", "Voice guide"),
    "core_traits": ("Traits", "Voice guide"),
    "micro_behaviors": ("Behavior", "Voice guide"),
    "vocabulary": ("Vocabulary", "Voice guide"),
    "preferences": ("Preferences", "Voice guide"),
    "relationships": ("Relationships", "Profile narrative"),
}


def _similarity(distance: float) -> float:
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, 1.0 - d)


def _truncate(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if len(clean) <= limit:
        return clean
    cut = clean[:limit].rsplit(" ", 1)[0] or clean[:limit]
    return cut + "…"


def _strip_header(content: str) -> str:
    return _HEADER_RE.sub("", content or "", count=1).strip()


def _episode_fields(body: str):
    try:
        items = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(items, dict):
        items = [items]
    if not items or not isinstance(items[0], dict):
        return None
    first = items[0]
    return (
        first.get("title"),
        first.get("summary"),
        first.get("sources"),
    )


def build_card(doc, relevance: int, display_name: str) -> dict:
    group = (doc.metadata or {}).get("group", "unknown")
    section, source_type = GROUP_INFO.get(group, ("Passage", "Profile text"))
    body = _strip_header(doc.page_content)
    title = f"{display_name} — {section}"
    cited = None

    if group == "episodes":
        parsed = _episode_fields(body)
        if parsed:
            ep_title, summary, sources = parsed
            if ep_title:
                title = f"{display_name} — Episode: {ep_title}"
            if summary:
                body = summary
            if isinstance(sources, list) and sources:
                cited = ", ".join(str(s) for s in sources[:3])

    return {
        "title": title,
        "source_type": source_type,
        "relevance": relevance,
        "preview": _truncate(body, PREVIEW_MAX),
        "excerpt": _truncate(body, EXCERPT_MAX),
        "meta": {
            k: v for k, v in {"section": section, "cited": cited}.items() if v
        },
    }


def build_evidence(results, elapsed_ms: float, display_name: str) -> dict:
    sims = [_similarity(dist) for _, dist in results]
    peak = max(sims) if sims else 0.0
    chunks = []
    for (doc, _), sim in zip(results, sims):
        relevance = round(100 * sim / peak) if peak > 0 else 0
        relevance = max(5, min(100, relevance)) if peak > 0 else 0
        chunks.append(build_card(doc, relevance, display_name))
    scores = [c["relevance"] for c in chunks]
    return {
        "chunks": chunks,
        "count": len(chunks),
        "best": max(scores) if scores else 0,
        "avg": round(sum(scores) / len(scores)) if scores else 0,
        "retrieval_ms": int(round(elapsed_ms)),
    }
