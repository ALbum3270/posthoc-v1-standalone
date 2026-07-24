"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Any, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from open_deep_research.graphrag.schemas import EvidencePack, GapStatus, GraphWriteResult

# Scalar state values and their defaults.
#
# These live here rather than as `field: float = 0.0` in the TypedDict body,
# because that form does not do what it appears to. A TypedDict is a plain dict
# at runtime, and a class-level assignment in its body is discarded -- it creates
# no default. Fields carrying a reducer get an initial value from their LangGraph
# channel, but plain scalars do not, so reading one before anything wrote it
# raises KeyError on the first round.
#
# Read them through `graph_state_value` (or `.get`) instead of subscripting.
GRAPH_STATE_DEFAULTS: dict[str, Any] = {
    "answer_confidence": 0.0,
    "coverage_ratio": 0.0,
    "rounds_without_improvement": 0,
}


def graph_state_value(state: Any, key: str) -> Any:
    """Read a scalar state value, falling back to its documented default."""

    if key not in GRAPH_STATE_DEFAULTS:
        raise KeyError(f"{key!r} has no declared default; add it to GRAPH_STATE_DEFAULTS")
    if state is None:
        return GRAPH_STATE_DEFAULTS[key]
    value = state.get(key, None)
    return GRAPH_STATE_DEFAULTS[key] if value is None else value


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing both legacy notes and graph-driven metadata."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    topic: Optional[str]
    current_focus_node_uuids: Annotated[list[str], override_reducer]
    pending_questions: Annotated[list[str], override_reducer]
    unresolved_conflict_ids: Annotated[list[str], override_reducer]
    gap_status: Annotated[list[GapStatus], override_reducer]
    graph_write_results: Annotated[list[GraphWriteResult], override_reducer]
    # NotRequired, not `= 0.0`: see GRAPH_STATE_DEFAULTS. Read via
    # graph_state_value(state, ...) so the first round does not KeyError.
    answer_confidence: NotRequired[float]
    coverage_ratio: NotRequired[float]
    rounds_without_improvement: NotRequired[int]
    evidence_pack: NotRequired[Optional[EvidencePack]]
    research_id: NotRequired[str]
    research_metrics: NotRequired[dict[str, Any]]
    raw_notes: Annotated[list[str], override_reducer]
    notes: Annotated[list[str], override_reducer]
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    topic: str
    current_focus_node_uuids: Annotated[list[str], override_reducer]
    pending_questions: Annotated[list[str], override_reducer]
    unresolved_conflict_ids: Annotated[list[str], override_reducer]
    gap_status: Annotated[list[GapStatus], override_reducer]
    graph_write_results: Annotated[list[GraphWriteResult], override_reducer]
    answer_confidence: float
    coverage_ratio: float
    rounds_without_improvement: int
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    topic: str
    current_focus_node_uuids: Annotated[list[str], override_reducer]
    pending_questions: Annotated[list[str], override_reducer]
    unresolved_conflict_ids: Annotated[list[str], override_reducer]
    graph_write_results: Annotated[list[GraphWriteResult], override_reducer]
    coverage_ratio: float
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    graph_write_results: list[GraphWriteResult] = Field(default_factory=list)


class ResearchState(TypedDict):
    """Lightweight graph-first state intended for the GraphRAG control loop."""

    topic: str
    research_brief: str
    current_focus_node_uuids: Annotated[list[str], override_reducer]
    pending_questions: Annotated[list[str], override_reducer]
    unresolved_conflict_ids: Annotated[list[str], override_reducer]
    gap_status: Annotated[list[GapStatus], override_reducer]
    graph_write_results: Annotated[list[GraphWriteResult], override_reducer]
    answer_confidence: float
    coverage_ratio: float
    rounds_without_improvement: int
    evidence_pack: EvidencePack | None
