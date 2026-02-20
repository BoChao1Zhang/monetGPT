"""
StateGraph construction for MonetAgent.

Builds the LangGraph graph:
  START → planner → executor → quality → human_review → context_fold → END
with conditional routing and error handling.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    planner_node,
    executor_node,
    quality_node,
    human_review_node,
    context_fold_node,
    error_handler_node,
    route_after_planner,
    route_after_quality,
    route_after_human,
)


def build_agent_graph(checkpointer=None, config: dict | None = None):
    """
    Build and compile the MonetAgent StateGraph.

    Args:
        checkpointer: LangGraph checkpointer (default: MemorySaver).
                       Pass SqliteSaver for persistent sessions.
        config: Optional config dict (reserved for future node factories).

    Returns:
        Compiled LangGraph graph ready for invoke/stream.
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("quality", quality_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("context_fold", context_fold_node)
    builder.add_node("error_handler", error_handler_node)

    # Edges
    builder.add_edge(START, "planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "executor": "executor",
            "context_fold": "context_fold",
            "error_handler": "error_handler",
        },
    )

    builder.add_edge("executor", "quality")

    builder.add_conditional_edges(
        "quality",
        route_after_quality,
        {
            "human_review": "human_review",
            "planner": "planner",
            "context_fold": "context_fold",
        },
    )

    builder.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "context_fold": "context_fold",
            "planner": "planner",
            "executor": "executor",
        },
    )

    builder.add_edge("context_fold", END)
    builder.add_edge("error_handler", "planner")

    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)
