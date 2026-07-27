"""Research harness: read full sources, decide the next step, verify afterwards.

The harness deliberately keeps every quality judgement out of the collection
path. Code owns the tools, the budget, mechanical comparison and bookkeeping;
the model owns every judgement. Nothing is rejected before it is recorded.
"""

from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ChecklistStatus,
    ResearchChecklist,
    build_checklist_prompt,
    generate_checklist,
)
from open_deep_research.harness.ledger import (
    ChecklistChangeRecord,
    ResearchLedger,
    RoundRecord,
)
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    QuoteSpan,
    ResearchNote,
    create_note,
)
from open_deep_research.harness.tools import (
    SearchResult,
    SourceReadError,
    read,
    search,
)

__all__ = [
    "ChecklistChangeRecord",
    "ChecklistDimension",
    "ChecklistItem",
    "ChecklistStatus",
    "NoteLocationStatus",
    "QuoteSpan",
    "ResearchChecklist",
    "ResearchLedger",
    "ResearchNote",
    "RoundRecord",
    "SearchResult",
    "SourceReadError",
    "build_checklist_prompt",
    "create_note",
    "generate_checklist",
    "read",
    "search",
]
