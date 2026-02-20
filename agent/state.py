"""
Core data structures for the MonetAgent hierarchical editing system.

AgentState is the LangGraph TypedDict that flows through all nodes.
Supporting dataclasses represent image versions, tool calls, action history, and sub-goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, TypedDict


# ---------------------------------------------------------------------------
# Reducer helper: append-only list (LangGraph accumulates instead of replacing)
# ---------------------------------------------------------------------------

def _append_reducer(existing: list, new: list) -> list:
    """LangGraph reducer: append new items to existing list."""
    if existing is None:
        existing = []
    if new is None:
        return existing
    return existing + new


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AssetNode:
    """Image version DAG node — tracks lineage of every intermediate image."""
    uri: str                    # file path of this image version
    parent_uri: str             # file path of the parent image (empty for original)
    transform_summary: str      # human-readable summary of what was done
    adjustments: dict = field(default_factory=dict)  # raw adjustments applied
    turn_id: int = 0
    step_id: int = -1


@dataclass
class ToolContext:
    """Ephemeral tool record — cleared after each turn's context fold."""
    tool_name: str              # e.g. "execute_single_stage"
    params: dict = field(default_factory=dict)
    thought: str = ""
    result_summary: str = ""
    success: bool = True


@dataclass
class ActionRecord:
    """Persistent high-level record — accumulated across turns."""
    turn_id: int
    intent: str                 # user's original request for this turn
    plan_summary: str           # what planner decided
    outcome: str                # "completed" | "partial" | "rolled_back"
    validated_asset_uri: str    # image path after quality check


@dataclass
class SubGoal:
    """Planner output: one discrete editing sub-goal."""
    id: int
    stage_type: str             # "global" | "local"
    operation_category: str     # "white-balance-tone-contrast" | "color-temperature" | "hsl" | "local-editing"
    description: str
    adjustments: dict = field(default_factory=dict)    # global ops: {op_name: value}
    local_specs: list = field(default_factory=list)    # local ops: JSON array of edit specs
    status: str = "pending"     # "pending" | "completed" | "failed"
    retry_count: int = 0


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Complete state flowing through the LangGraph StateGraph."""
    # Session identity
    session_id: str
    original_image_path: str
    current_image_path: str
    style: str

    # Turn tracking
    turn_id: int
    user_message: str

    # Planner output
    sub_goals: list                                         # List[SubGoal]
    current_sub_goal_idx: int
    plan_version: int
    replan_attempts: int

    # Pipeline state carried across stages
    accrued_dehaze: float

    # Quality assessment
    quality_score: float
    quality_pass: bool
    quality_assessment: str

    # Image version graph (kept bounded by prune_asset_graph)
    asset_graph: list                                       # List[AssetNode]

    # Ephemeral tool records (cleared each fold)
    tool_contexts: list                                     # List[ToolContext]

    # Persistent action history (append-only)
    action_history: Annotated[list, _append_reducer]        # List[ActionRecord]

    # Human-in-the-loop
    human_decision: str                                     # approve | modify | rollback | replan
    human_modifications: dict
    review_request: dict
    rollback_target: dict

    # Error handling
    error_message: str

    # Termination
    is_complete: bool

    # Runtime config snapshot (loaded from configs/agent_config.yaml)
    agent_config: dict
