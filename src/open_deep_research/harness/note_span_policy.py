"""Mechanical capacity policy for model-selected note source ranges."""

from __future__ import annotations

from enum import Enum

from open_deep_research.harness.source_spans import ResolvedSourceSpan

# These are provisional protocol-capacity ceilings, not evidence-quality
# thresholds. They freeze the first observed separation between compact ranges
# (at most 9 segments / 1,185 characters) and section-scale selections (at
# least 24 segments / 2,416 characters). A future change must be explicit; code
# never truncates a proposed range to make it fit.
DEFAULT_NOTE_SPAN_MAX_SEGMENTS = 12
DEFAULT_NOTE_SPAN_MAX_CHARS = 2_000


class SourceSpanCapacityReason(str, Enum):
    """Mechanical reasons a resolved pointer range exceeds its envelope."""

    TOO_MANY_SEGMENTS = "span_too_many_segments"
    TOO_MANY_CHARS = "span_too_many_chars"


class SourceSpanCapacityError(ValueError):
    """One fully resolved range rejected without truncation or rewriting."""

    def __init__(
        self,
        *,
        start_segment_id: str,
        end_segment_id: str,
        segment_count: int,
        char_count: int,
        max_segments: int,
        max_chars: int,
        failure_reasons: tuple[SourceSpanCapacityReason, ...],
    ) -> None:
        self.start_segment_id = start_segment_id
        self.end_segment_id = end_segment_id
        self.segment_count = segment_count
        self.char_count = char_count
        self.max_segments = max_segments
        self.max_chars = max_chars
        self.failure_reasons = failure_reasons
        reason_text = ", ".join(reason.value for reason in failure_reasons)
        super().__init__(
            f"source span exceeds protocol capacity ({reason_text}); "
            f"segments={segment_count}/{max_segments}, "
            f"chars={char_count}/{max_chars}"
        )

    def audit_payload(self) -> dict[str, object]:
        """Return the complete rejected proposal for durable audit."""

        return {
            "start_segment_id": self.start_segment_id,
            "end_segment_id": self.end_segment_id,
            "segment_count": self.segment_count,
            "char_count": self.char_count,
            "max_segments": self.max_segments,
            "max_chars": self.max_chars,
            "failure_reasons": [
                reason.value for reason in self.failure_reasons
            ],
        }


def enforce_source_span_capacity(
    resolved: ResolvedSourceSpan,
    *,
    max_segments: int = DEFAULT_NOTE_SPAN_MAX_SEGMENTS,
    max_chars: int = DEFAULT_NOTE_SPAN_MAX_CHARS,
) -> ResolvedSourceSpan:
    """Reject an oversized range whole; never clamp, truncate, or split it."""

    if max_segments < 1:
        raise ValueError("max_segments must be at least 1")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    reasons: list[SourceSpanCapacityReason] = []
    if resolved.segment_count > max_segments:
        reasons.append(SourceSpanCapacityReason.TOO_MANY_SEGMENTS)
    char_count = len(resolved.source_quote)
    if char_count > max_chars:
        reasons.append(SourceSpanCapacityReason.TOO_MANY_CHARS)
    if reasons:
        raise SourceSpanCapacityError(
            start_segment_id=resolved.start_segment_id,
            end_segment_id=resolved.end_segment_id,
            segment_count=resolved.segment_count,
            char_count=char_count,
            max_segments=max_segments,
            max_chars=max_chars,
            failure_reasons=tuple(reasons),
        )
    return resolved
