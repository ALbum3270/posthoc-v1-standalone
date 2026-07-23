"""Targeted extraction driven by ontology gaps rather than full-document NER."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.configuration import Configuration
from open_deep_research.graphrag.ontology import OntologySlot, get_slot
from open_deep_research.graphrag.prompts import (
    TARGETED_EXTRACTION_SYSTEM_PROMPT,
    build_targeted_extraction_user_prompt,
)
from open_deep_research.graphrag.schemas import (
    EntityRef,
    ExtractedClaim,
    ExtractedTriple,
    SourceDocument,
    SourceSpan,
)


class TargetedExtractionConfig(BaseModel):
    """Runtime settings for targeted extraction."""

    model_config = ConfigDict(extra="forbid")

    model: str = "openai:gpt-4.1-mini"
    max_output_tokens: int = 4096
    max_chars: int = 4000
    max_retries: int = 3


class _ExtractionTriple(BaseModel):
    """Structured LLM output for one extracted claim."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    claim_text: str
    subject_name: str
    subject_type: str = "Entity"
    predicate: str
    object_value: str
    object_type: str = "Literal"
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None
    quote: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class _TargetedExtractionResponse(BaseModel):
    """Structured LLM response for targeted extraction."""

    model_config = ConfigDict(extra="forbid")

    claims: list[_ExtractionTriple] = Field(default_factory=list)


class TargetedExtractor:
    """Extract only facts that map to open ontology slots."""

    def __init__(
        self,
        config: TargetedExtractionConfig,
        *,
        model: Any | None = None,
    ) -> None:
        self.config = config
        self._model = model

    @classmethod
    def from_runnable_config(
        cls,
        config: RunnableConfig | None = None,
        *,
        model: Any | None = None,
    ) -> "TargetedExtractor":
        """Create an extractor from the app-level configuration."""

        app_config = Configuration.from_runnable_config(config)
        extraction_config = TargetedExtractionConfig(
            model=app_config.targeted_extraction_model,
            max_output_tokens=app_config.targeted_extraction_model_max_tokens,
            max_chars=app_config.targeted_extraction_max_chars,
            max_retries=app_config.max_structured_output_retries,
        )
        return cls(extraction_config, model=model)

    async def extract(
        self,
        *,
        topic: str,
        document: SourceDocument,
        pending_questions: list[str],
        runnable_config: RunnableConfig | None = None,
        schema: dict[str, tuple[OntologySlot, ...]] | None = None,
    ) -> list[ExtractedClaim]:
        """Extract claims that fill the requested ontology gaps."""

        slots = self._resolve_pending_slots(pending_questions, schema=schema)
        if not slots:
            return []

        extractor_model = self._get_model(runnable_config)
        prompt = build_targeted_extraction_user_prompt(
            topic=topic,
            document=document,
            slots=slots,
            max_chars=self.config.max_chars,
        )
        response = await extractor_model.ainvoke(
            [
                SystemMessage(content=TARGETED_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return self._map_response_to_claims(
            response=response,
            document=document,
            allowed_slot_ids={slot.slot_id for slot in slots},
        )

    def _get_model(self, runnable_config: RunnableConfig | None) -> Any:
        """Create or return the configured structured-output model."""

        if self._model is not None:
            return self._model

        from langchain.chat_models import init_chat_model

        model = init_chat_model(
            model=self.config.model,
            max_tokens=self.config.max_output_tokens,
            api_key=self._get_api_key_for_model(self.config.model, runnable_config),
            tags=["langsmith:nostream"],
        )
        return model.with_structured_output(_TargetedExtractionResponse).with_retry(
            stop_after_attempt=self.config.max_retries
        )

    @staticmethod
    def _resolve_pending_slots(
        pending_questions: list[str],
        *,
        schema: dict[str, tuple[OntologySlot, ...]] | None = None,
    ) -> list[OntologySlot]:
        """Resolve pending questions to concrete ontology slots."""

        resolved: list[OntologySlot] = []
        seen: set[str] = set()

        for item in pending_questions:
            slot = get_slot(item, schema)
            if slot is None:
                slot = TargetedExtractor._lookup_slot_by_label_or_question(item, schema)
            if slot is None or slot.slot_id in seen:
                continue
            resolved.append(slot)
            seen.add(slot.slot_id)

        return resolved

    @staticmethod
    def _lookup_slot_by_label_or_question(
        text: str,
        schema: dict[str, tuple[OntologySlot, ...]] | None = None,
    ) -> OntologySlot | None:
        """Resolve slot references by label, alias, or question text."""

        normalized = text.strip().casefold()
        if not normalized:
            return None

        active_schema = schema or {}
        if not active_schema:
            from open_deep_research.graphrag.ontology import INVESTIGATION_SCHEMA, iter_slots

            slots = iter_slots(INVESTIGATION_SCHEMA)
        else:
            from open_deep_research.graphrag.ontology import iter_slots

            slots = iter_slots(active_schema)

        for slot in slots:
            candidates = [slot.slot_id, slot.label, slot.question, *slot.aliases]
            if any(candidate.strip().casefold() == normalized for candidate in candidates):
                return slot
        return None

    @staticmethod
    def _get_api_key_for_model(
        model_name: str,
        runnable_config: RunnableConfig | None,
    ) -> str | None:
        """Local API key lookup to avoid importing the broader utils module."""

        should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
        normalized_model_name = model_name.lower()

        if should_get_from_config.lower() == "true":
            api_keys = (runnable_config or {}).get("configurable", {}).get("apiKeys", {})
            if not api_keys:
                return None
            if normalized_model_name.startswith("openai:"):
                return api_keys.get("OPENAI_API_KEY")
            if normalized_model_name.startswith("anthropic:"):
                return api_keys.get("ANTHROPIC_API_KEY")
            if normalized_model_name.startswith("google"):
                return api_keys.get("GOOGLE_API_KEY")
            return None

        if normalized_model_name.startswith("openai:"):
            return os.getenv("OPENAI_API_KEY")
        if normalized_model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        if normalized_model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

    @staticmethod
    def _map_response_to_claims(
        *,
        response: _TargetedExtractionResponse,
        document: SourceDocument,
        allowed_slot_ids: set[str],
    ) -> list[ExtractedClaim]:
        """Map structured LLM output to shared extraction schemas."""

        mapped_claims: list[ExtractedClaim] = []

        for index, item in enumerate(response.claims):
            if item.slot_id not in allowed_slot_ids:
                continue

            span = None
            if item.start_char is not None and item.end_char is not None:
                span = SourceSpan(
                    start_char=item.start_char,
                    end_char=item.end_char,
                    quote=item.quote,
                )

            object_ref: EntityRef | str
            if item.object_type.lower() == "entity":
                object_ref = EntityRef(
                    name=item.object_value,
                    entity_type="Entity",
                )
            else:
                object_ref = item.object_value

            triple = ExtractedTriple(
                slot_id=item.slot_id,
                subject=EntityRef(
                    name=item.subject_name,
                    entity_type=item.subject_type,
                ),
                predicate=item.predicate,
                object=object_ref,
                confidence=item.confidence,
                source_document_id=document.document_id,
                source_span=span,
                rationale=item.rationale,
            )
            mapped_claims.append(
                ExtractedClaim(
                    claim_id=f"{document.document_id}:{item.slot_id}:{index}",
                    slot_id=item.slot_id,
                    text=item.claim_text,
                    triples=[triple],
                    source_document_id=document.document_id,
                    source_span=span,
                    confidence=item.confidence,
                    metadata={
                        "document_title": document.title,
                        "document_url": document.url,
                    },
                )
            )

        return mapped_claims
