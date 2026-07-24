"""Turn a raw page into the passages worth extracting from.

§3.11 constraint 2. Naive head truncation -- ``raw_text[:2000]`` -- is what
starved extraction in the V1 baseline: a Wikipedia page arrives as 145,402
characters whose first two thousand are navigation chrome, so the model read
``[Jump to content]`` and image paths and returned nothing, round after round.

Three stages, all deterministic:

1. **clean** -- drop markup and known navigation lines, keep paragraphs;
2. **chunk** -- split on paragraph boundaries, not character offsets;
3. **select** -- rank chunks against the slot question and keep the best.

Selecting by relevance rather than position is the point. The passage answering
"what was the scale of the losses" is rarely in the first 2,000 characters, and
on a long page it is usually nowhere near them.
"""

from __future__ import annotations

import re

_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t]+")
_BLANK_RUN = re.compile(r"\n{3,}")
_TOKEN = re.compile(r"[a-z0-9]{2,}|[一-鿿]")

# Everything from one of these headings to the end of the page is apparatus,
# not content. Measured on the 2026-07-24 regression: 3 of 105 evidence items
# were bibliography entries, and *all three* landed in WHEN slots -- a citation's
# "Retrieved 19 July 2024" reads exactly like an event date, and one of them was
# the evidence that made the CrowdStrike date check pass. Small in count,
# concentrated where it does the most damage.
_REFERENCE_HEADINGS = (
    "references",
    "reference list",
    "notes",
    "footnotes",
    "citations",
    "bibliography",
    "further reading",
    "external links",
    "see also",
    "sources",
    "参考文献",
    "参考资料",
    "参见",
    "外部链接",
    "延伸阅读",
    "注释",
    "脚注",
)
# Citation-shaped lines that appear inline rather than under a heading.
_CITATION_LINE = re.compile(
    r"Retrieved\s+\d|Archived\s+from\s+the\s+original|\bISBN\b|\bdoi:|\barXiv:"
    r"|\bVol\.\s|\bpp?\.\s?\d+|\(PDF\)",
    re.IGNORECASE,
)

_CHROME_LINES = {
    "jump to content",
    "jump to navigation",
    "main menu",
    "navigation menu",
    "skip to main content",
    "table of contents",
    "search",
    "edit",
    "cookie policy",
}

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are",
    "what", "who", "when", "where", "why", "how", "did", "does", "has", "have",
    "its", "his", "her", "their", "about", "into", "than", "then", "been",
    "of", "in", "on", "at", "to", "by", "is", "it", "as", "an", "or", "be",
}


def _is_reference_heading(line: str) -> bool:
    """Whether a line is the heading that starts a page's citation apparatus.

    Matched conservatively: a heading is short and consists of the marker alone,
    so a sentence merely containing the word "sources" is not mistaken for one.
    """

    stripped = line.strip().strip("#*=_-· ").casefold()
    if not stripped or len(stripped) > 24:
        return False
    return stripped in _REFERENCE_HEADINGS


def clean_text(text: str) -> str:
    """Strip markup and navigation furniture, preserving paragraph structure."""

    if not text:
        return ""

    cleaned = _MARKDOWN_IMAGE.sub(" ", text)
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    lines: list[str] = []
    for line in cleaned.split("\n"):
        normalized = _SPACES.sub(" ", line).strip()
        if not normalized:
            lines.append("")
            continue
        if normalized.casefold() in _CHROME_LINES:
            continue
        if _is_reference_heading(normalized):
            # Apparatus starts here; nothing below it is the article's own claim.
            break
        if _CITATION_LINE.search(normalized):
            continue
        lines.append(normalized)

    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def query_terms(*parts: str) -> set[str]:
    """Content words from the slot question, for scoring. Handles CJK and latin."""

    terms: set[str] = set()
    for part in parts:
        for token in _TOKEN.findall((part or "").casefold()):
            if token not in _STOPWORDS:
                terms.add(token)
    return terms


def split_chunks(text: str, *, target_chars: int = 900) -> list[str]:
    """Split on paragraph boundaries, packing up to ``target_chars`` per chunk."""

    chunks: list[str] = []
    buffer: list[str] = []
    size = 0

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # A single oversized paragraph still has to be broken somewhere.
        pieces = [
            paragraph[i : i + target_chars]
            for i in range(0, len(paragraph), target_chars)
        ] or [paragraph]
        for piece in pieces:
            if buffer and size + len(piece) > target_chars:
                chunks.append("\n\n".join(buffer))
                buffer, size = [], 0
            buffer.append(piece)
            size += len(piece)

    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def score_chunk(chunk: str, terms: set[str]) -> float:
    """Rank a chunk: question terms dominate, prose is preferred to link soup."""

    lowered = chunk.casefold()
    matched = sum(min(lowered.count(term), 3) for term in terms)
    numbers = len(re.findall(r"\b\d[\d,.%$€£¥-]*", chunk))
    prose = len(re.findall(r"[A-Za-z一-鿿]", chunk))
    link_noise = lowered.count("http") + lowered.count(".svg") + lowered.count(".png")

    score = matched * 4.0
    score += min(numbers, 8) * 0.15
    score += min(prose / max(len(chunk), 1), 1.0)
    score -= link_noise * 0.5
    return score


def select_relevant_text(
    raw_text: str,
    *,
    focus: str = "",
    max_chars: int = 2000,
    target_chunk_chars: int = 900,
) -> str:
    """Return the passages most relevant to ``focus``, capped at ``max_chars``.

    Falls back to the cleaned head of the document when there is no focus or
    nothing scores -- still better than raw truncation, since chrome is gone.
    Selected chunks are re-emitted in document order so the passage reads
    naturally rather than in score order.
    """

    cleaned = clean_text(raw_text)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned

    # A chunk larger than the budget can never be selected, which would silently
    # degrade selection back to "return the head". Keep chunks affordable.
    chunk_size = min(target_chunk_chars, max(max_chars // 2, 120))
    chunks = split_chunks(cleaned, target_chars=chunk_size)
    if not chunks:
        return cleaned[:max_chars]

    terms = query_terms(focus)
    if not terms:
        return cleaned[:max_chars]

    ranked = sorted(
        ((score_chunk(chunk, terms), index) for index, chunk in enumerate(chunks)),
        key=lambda pair: (-pair[0], pair[1]),
    )

    chosen: list[int] = []
    used = 0
    for score, index in ranked:
        if score <= 0 and chosen:
            break
        length = len(chunks[index]) + (2 if chosen else 0)
        if used + length > max_chars:
            continue
        chosen.append(index)
        used += length

    if not chosen:
        return cleaned[:max_chars]

    return "\n\n".join(chunks[index] for index in sorted(chosen))
