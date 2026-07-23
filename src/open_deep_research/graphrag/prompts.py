"""Prompt templates for graph-driven extraction and reasoning."""

from __future__ import annotations

from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.schemas import SourceDocument


TARGETED_EXTRACTION_SYSTEM_PROMPT = """You perform targeted information extraction for a graph-based research agent.

You must only extract information that directly helps fill the requested investigation slots.
Ignore all unrelated facts, speculation, marketing language, and background detail.

Return structured output that obeys the response schema exactly.

Extraction rules:
- Prefer precise factual statements over vague summaries.
- Keep one claim atomic. Do not merge multiple facts into one claim.
- Keep slot_id exactly equal to one of the requested slot ids.
- Use confidence between 0 and 1.
- If the evidence is weak, either lower confidence or omit the claim.
- Copy short quotes only when they directly justify the claim.
- Use object_type "Literal" for dates, amounts, places, and other literal values when appropriate.
- Use object_type "Entity" when the object is another named actor, organization, product, or place.
"""


def build_targeted_extraction_user_prompt(
    *,
    topic: str,
    document: SourceDocument,
    slots: list[OntologySlot],
    max_chars: int,
) -> str:
    """Build the user prompt for targeted extraction."""

    slot_lines = "\n".join(
        f"- {slot.slot_id} | {slot.label} | {slot.question}"
        for slot in slots
    )

    content = document.content[:max_chars]

    return f"""Topic:
{topic}

Requested investigation slots:
{slot_lines}

Document metadata:
- document_id: {document.document_id}
- title: {document.title}
- url: {document.url or ""}
- source_type: {document.source_type.value}

Document content:
\"\"\"{content}\"\"\"

Extract only facts that directly fill the requested slots."""
