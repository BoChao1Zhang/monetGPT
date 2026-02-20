"""
MonetAgent: Hierarchical Agent system for multi-turn image editing.

Built on LangGraph, wraps the existing MonetGPT pipeline (ImageEditingPipeline + MaskedExecutor)
with Planner → Executor → Quality → HumanReview → ContextFold loop.
"""

from .state import AgentState, AssetNode, ToolContext, ActionRecord, SubGoal
from .session import create_checkpointer, create_session, load_agent_config

try:
    from .graph import build_agent_graph
except Exception as _graph_import_error:  # pragma: no cover - only for minimal envs
    def build_agent_graph(*args, **kwargs):
        raise ImportError(
            "build_agent_graph requires langgraph to be installed."
        ) from _graph_import_error

__all__ = [
    "AgentState",
    "AssetNode",
    "ToolContext",
    "ActionRecord",
    "SubGoal",
    "build_agent_graph",
    "create_session",
    "create_checkpointer",
    "load_agent_config",
]
