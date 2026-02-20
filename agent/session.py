"""
Session management for MonetAgent.

Creates session directories, initialises AgentState, and manages session lifecycle.
"""

from __future__ import annotations

import os
import shutil
import uuid
import atexit
from typing import Optional

import yaml

from .state import AgentState, AssetNode
from dataclasses import asdict

DEFAULT_AGENT_CONFIG = {
    "planner": {
        "max_sub_goals": 6,
        "temperature": 0.2,
        "max_retries_per_sub_goal": 2,
    },
    "quality": {
        "min_pass_score": 0.6,
        "temperature": 0.1,
        "auto_approve_threshold": 0.85,
    },
    "context_folding": {
        "max_history_tokens": 2000,
        "keep_recent_turns": 3,
    },
    "session": {
        "base_dir": "./sessions",
        "checkpoint_backend": "memory",
        "sqlite_path": "./sessions/checkpoints.db",
        "output_extension": ".png",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_agent_config(config_path: str = "configs/agent_config.yaml") -> dict:
    """Load agent configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_AGENT_CONFIG, loaded)


def create_checkpointer(config: dict):
    """
    Build checkpointer from agent config.

    Supported backends:
      - memory (default)
      - sqlite (requires langgraph checkpoint sqlite extra)
    """
    backend = str(config.get("session", {}).get("checkpoint_backend", "memory")).lower()
    if backend == "sqlite":
        sqlite_path = config.get("session", {}).get("sqlite_path", "./sessions/checkpoints.db")
        os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            # Support both raw path and sqlite URI depending on installed version.
            try:
                saver_or_cm = SqliteSaver.from_conn_string(sqlite_path)
            except Exception:
                saver_or_cm = SqliteSaver.from_conn_string(f"sqlite:///{sqlite_path}")

            # Some langgraph versions return a context manager; enter once and
            # keep it alive for the process lifetime.
            if hasattr(saver_or_cm, "__enter__") and hasattr(saver_or_cm, "__exit__"):
                cm = saver_or_cm
                saver = cm.__enter__()

                def _close_sqlite_cm():
                    try:
                        cm.__exit__(None, None, None)
                    except Exception:
                        pass

                atexit.register(_close_sqlite_cm)
                return saver
            return saver_or_cm
        except Exception as exc:
            print(f"[Session] SqliteSaver unavailable, fallback to MemorySaver: {exc}")

    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def create_session(
    image_path: str,
    style: str = "balanced",
    config_path: str = "configs/agent_config.yaml",
    session_id: Optional[str] = None,
) -> tuple[AgentState, str]:
    """
    Initialise a new editing session.

    Args:
        image_path: Path to the source image.
        style: Editing style ("balanced", "vibrant", "retro").
        config_path: Path to agent config YAML.
        session_id: Optional explicit session ID (default: auto-generated UUID).

    Returns:
        Tuple of (initial AgentState dict, session_id).
    """
    config = load_agent_config(config_path)
    base_dir = config.get("session", {}).get("base_dir", "./sessions")

    if session_id is None:
        session_id = uuid.uuid4().hex[:12]

    session_dir = os.path.join(base_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Copy original image into session directory
    ext = os.path.splitext(image_path)[1]
    original_copy = os.path.join(session_dir, f"original{ext}")
    shutil.copy2(image_path, original_copy)

    original_asset = AssetNode(
        uri=original_copy,
        parent_uri="",
        transform_summary="Original image",
        adjustments={},
        turn_id=0,
        step_id=-1,
    )

    state: AgentState = {
        "session_id": session_id,
        "original_image_path": original_copy,
        "current_image_path": original_copy,
        "style": style,
        "turn_id": 0,
        "user_message": "",
        "sub_goals": [],
        "current_sub_goal_idx": 0,
        "plan_version": 0,
        "replan_attempts": 0,
        "accrued_dehaze": 0.0,
        "quality_score": 0.0,
        "quality_pass": False,
        "quality_assessment": "",
        "asset_graph": [asdict(original_asset)],
        "tool_contexts": [],
        "action_history": [],
        "human_decision": "",
        "human_modifications": {},
        "review_request": {},
        "rollback_target": {},
        "error_message": "",
        "is_complete": False,
        "agent_config": config,
    }

    return state, session_id
