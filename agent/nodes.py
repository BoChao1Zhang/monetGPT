"""
Node implementations for the MonetAgent StateGraph.

Nodes: planner, executor, quality, human_review, context_fold, error_handler.
Each node takes AgentState and returns a partial state update dict.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import asdict
from typing import Any
from .state import AgentState, AssetNode, ToolContext, SubGoal
from .prompts import (
    PLANNER_SYSTEM_PROMPT,
    QUALITY_SYSTEM_PROMPT,
    _create_planner_prompt,
    _create_quality_prompt,
)
from .context_folding import fold_context_for_prompt, prune_asset_graph


# ---------------------------------------------------------------------------
# Lazy singletons (avoid importing heavy modules at import time)
# ---------------------------------------------------------------------------

_inference_engine = None
_image_pipeline = None


def _get_inference_engine():
    global _inference_engine
    if _inference_engine is None:
        from inference.core import InferenceEngine
        _inference_engine = InferenceEngine()
    return _inference_engine


def _get_image_pipeline():
    global _image_pipeline
    if _image_pipeline is None:
        from pipeline.core import ImageEditingPipeline
        _image_pipeline = ImageEditingPipeline()
    return _image_pipeline


def warmup_local_editing_models():
    """
    Eagerly load GroundingDINO + SAM2 once for lower interactive latency.
    Returns (ok, message).
    """
    try:
        pipeline = _get_image_pipeline()
        pipeline.warmup_local_models()
        return True, "Local models warmed up."
    except Exception as e:
        return False, f"Local model warmup failed: {e}"


# ---------------------------------------------------------------------------
# 1. Planner Node
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """
    Decompose user request into SubGoals via VLM.

    Reads: user_message, current_image_path, style, asset_graph, action_history,
           quality_assessment (if replanning)
    Writes: sub_goals, current_sub_goal_idx, error_message
    """
    try:
        engine = _get_inference_engine()

        # Build compressed history
        keep_recent_turns = int(_cfg(state, "context_folding", "keep_recent_turns", default=3))
        max_history_tokens = int(_cfg(state, "context_folding", "max_history_tokens", default=2000))
        folded = fold_context_for_prompt(
            state.get("asset_graph", []),
            state.get("action_history", []),
            state.get("current_image_path", ""),
            keep_recent=keep_recent_turns,
            max_history_tokens=max_history_tokens,
        )

        # Build replan feedback if quality triggered a replan
        replan_feedback = ""
        if state.get("quality_assessment") and (
            not state.get("quality_pass", True) or state.get("human_decision") == "replan"
        ):
            replan_feedback = state["quality_assessment"]

        user_prompt = _create_planner_prompt(
            user_message=state.get("user_message", ""),
            style=state.get("style", "balanced"),
            folded_context=folded,
            replan_feedback=replan_feedback,
        )

        # VLM call
        response = engine.query_structured(
            image_path=state.get("current_image_path", ""),
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=float(_cfg(state, "planner", "temperature", default=0.2)),
        )

        # Parse SubGoals (primary: JSON array; fallback: adjustment dict/object)
        raw_goals = engine.extract_json_array_from_response(response)
        if not raw_goals:
            raw_goals = _fallback_parse_planner_response(engine, response)
        max_sub_goals = int(_cfg(state, "planner", "max_sub_goals", default=6))
        raw_goals = raw_goals[:max_sub_goals]
        sub_goals = []
        for g in raw_goals:
            sg = SubGoal(
                id=g.get("id", len(sub_goals) + 1),
                stage_type=g.get("stage_type", "global"),
                operation_category=g.get("operation_category", ""),
                description=g.get("description", ""),
                adjustments=g.get("adjustments", {}),
                local_specs=g.get("local_specs", []),
                status="pending",
                retry_count=0,
            )
            sub_goals.append(asdict(sg))

        print(f"[Planner] Generated {len(sub_goals)} sub-goals")
        for sg in sub_goals:
            print(f"  #{sg['id']}: [{sg['stage_type']}] {sg['description']}")

        return {
            "sub_goals": sub_goals,
            "current_sub_goal_idx": 0,
            "plan_version": _next_plan_version(state),
            "replan_attempts": _next_replan_attempts(state),
            "human_decision": "",
            "human_modifications": {},
            "rollback_target": {},
            "error_message": "",
        }

    except Exception as e:
        print(f"[Planner] Error: {e}")
        traceback.print_exc()
        return {"error_message": f"Planner error: {e}", "sub_goals": []}


# ---------------------------------------------------------------------------
# 2. Executor Node
# ---------------------------------------------------------------------------

def executor_node(state: AgentState) -> dict:
    """
    Execute ALL pending sub-goals sequentially.

    For each sub-goal:
      - global → write temp config JSON → ImageEditingPipeline.execute_single_stage()
      - local  → write temp config JSON array → execute_single_stage(is_local=True)

    Writes: current_image_path, sub_goals (status updates), asset_graph, tool_contexts,
            accrued_dehaze, current_sub_goal_idx
    """
    pipeline = _get_image_pipeline()
    sub_goals = [dict(sg) for sg in state.get("sub_goals", [])]
    current_image = state.get("current_image_path", "")
    session_dir = os.path.dirname(current_image) if current_image else "./sessions/tmp"
    turn_id = state.get("turn_id", 0)
    accrued_dehaze = state.get("accrued_dehaze", 0.0)

    asset_graph = [dict(node) for node in state.get("asset_graph", [])]
    tool_contexts = [dict(ctx) for ctx in state.get("tool_contexts", [])]
    idx = state.get("current_sub_goal_idx", 0)

    while idx < len(sub_goals):
        sg = sub_goals[idx]
        if sg.get("status") != "pending":
            idx += 1
            continue

        output_ext = str(
            _cfg(state, "session", "output_extension", default=".png")
        ).strip() or ".png"
        if not output_ext.startswith("."):
            output_ext = f".{output_ext}"
        step_output = os.path.join(
            session_dir, f"turn{turn_id}_step{idx}{output_ext.lower()}"
        )

        print(f"[Executor] Step {idx}: {sg.get('description', '')}")

        try:
            applied_adjustments = sg.get("adjustments", {})
            if sg.get("stage_type") == "local":
                # Local editing path
                pipeline.execute_single_stage(
                    sg.get("local_specs", []),
                    current_image,
                    step_output,
                    is_local=True,
                )
            else:
                # Global editing path
                adjustments = sg.get("adjustments", {})

                # Apply value scaling for global stages
                stage_idx = _operation_category_to_stage(sg.get("operation_category", ""))
                if stage_idx is not None and adjustments:
                    from config import get_processed_predictions
                    adjustments, accrued_dehaze = get_processed_predictions(
                        stage_idx, adjustments, accrued_dehaze
                    )
                    if stage_idx == 2:
                        adjustments["Dehaze"] = int(accrued_dehaze)
                applied_adjustments = adjustments

                pipeline.execute_single_stage(
                    adjustments,
                    current_image,
                    step_output,
                    is_local=False,
                )

            # Record asset
            asset = AssetNode(
                uri=step_output,
                parent_uri=current_image,
                transform_summary=f"T{turn_id}S{idx}: {sg.get('description', '')}",
                adjustments=applied_adjustments if sg.get("stage_type") != "local" else sg.get("local_specs", []),
                turn_id=turn_id,
                step_id=idx,
            )
            asset_graph = _upsert_asset_node(asset_graph, asdict(asset))

            # Record tool context
            tc = ToolContext(
                tool_name="execute_single_stage",
                params={"stage_type": sg.get("stage_type"), "category": sg.get("operation_category")},
                thought=sg.get("description", ""),
                result_summary=f"Output: {step_output}",
                success=True,
            )
            tool_contexts.append(asdict(tc))

            sg["status"] = "completed"
            current_image = step_output

        except Exception as e:
            print(f"[Executor] Error on step {idx}: {e}")
            traceback.print_exc()
            sg["status"] = "failed"
            tc = ToolContext(
                tool_name="execute_single_stage",
                params={"stage_type": sg.get("stage_type")},
                thought=sg.get("description", ""),
                result_summary=f"Error: {e}",
                success=False,
            )
            tool_contexts.append(asdict(tc))

        idx += 1

    return {
        "current_image_path": current_image,
        "sub_goals": sub_goals,
        "current_sub_goal_idx": idx,
        "accrued_dehaze": accrued_dehaze,
        "asset_graph": asset_graph,
        "tool_contexts": tool_contexts,
    }


# ---------------------------------------------------------------------------
# 3. Quality Node
# ---------------------------------------------------------------------------

def quality_node(state: AgentState) -> dict:
    """
    Assess edit quality by sending original + current to VLM.

    Writes: quality_score, quality_pass, quality_assessment
    """
    try:
        engine = _get_inference_engine()

        # Build sub-goal descriptions
        sg_descs = []
        adj_parts = []
        for sg in state.get("sub_goals", []):
            sg_descs.append(f"[{sg.get('stage_type')}] {sg.get('description', '')}")
            if sg.get("adjustments"):
                adj_parts.append(json.dumps(sg["adjustments"]))
            if sg.get("local_specs"):
                adj_parts.append(f"{len(sg['local_specs'])} local edits")

        user_prompt = _create_quality_prompt(
            user_intent=state.get("user_message", ""),
            sub_goal_descriptions="\n".join(sg_descs),
            adjustments_summary="\n".join(adj_parts) if adj_parts else "N/A",
        )

        # VLM call with both images
        response = engine.query_structured(
            image_path=state.get("current_image_path", ""),
            system_prompt=QUALITY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=float(_cfg(state, "quality", "temperature", default=0.1)),
            original_image_path=state.get("original_image_path", ""),
        )

        # Parse quality response
        result = engine.extract_json_from_response(response)
        score = _safe_float(result.get("score"), 0.5)
        assessment = result.get("assessment", response)
        min_pass_score = float(_cfg(state, "quality", "min_pass_score", default=0.6))
        passed = score >= min_pass_score

        print(f"[Quality] Score: {score:.2f} {'✓' if passed else '✗'}")
        print(f"[Quality] {assessment}")

        return {
            "quality_score": score,
            "quality_pass": passed,
            "quality_assessment": assessment,
        }

    except Exception as e:
        print(f"[Quality] Error: {e}")
        traceback.print_exc()
        min_pass_score = float(_cfg(state, "quality", "min_pass_score", default=0.6))
        # On quality errors, route to human review/replan instead of silently passing.
        return {
            "quality_score": max(0.0, min_pass_score - 0.1),
            "quality_pass": False,
            "quality_assessment": f"Quality check error: {e}",
        }


# ---------------------------------------------------------------------------
# 4. Human Review Node
# ---------------------------------------------------------------------------

def human_review_node(state: AgentState) -> dict:
    """
    Pause execution for human review using LangGraph interrupt().

    Presents: current image, quality score, assessment, sub-goal summary.
    Expects Command(resume={"decision": ..., "modifications": ...}) from UI.

    Writes: human_decision, human_modifications
    """
    from langgraph.types import interrupt

    review_payload = {
        "current_image": state.get("current_image_path", ""),
        "original_image": state.get("original_image_path", ""),
        "quality_score": state.get("quality_score", 0.0),
        "quality_assessment": state.get("quality_assessment", ""),
        "sub_goals": [
            {"id": sg.get("id"), "description": sg.get("description"), "status": sg.get("status")}
            for sg in state.get("sub_goals", [])
        ],
        "turn_id": state.get("turn_id", 0),
        "plan_version": state.get("plan_version", 0),
    }

    # Interrupt and wait for human input
    human_input = interrupt(review_payload)

    decision = human_input.get("decision", "approve") if isinstance(human_input, dict) else "approve"
    decision = str(decision).lower()
    modifications = human_input.get("modifications", {}) if isinstance(human_input, dict) else {}
    rollback_target = human_input.get("rollback_target", {}) if isinstance(human_input, dict) else {}

    print(f"[HumanReview] Decision: {decision}")

    if decision == "rollback":
        rollback_uri, resolved_target = _resolve_rollback_uri(state, rollback_target)
        return {
            "human_decision": "rollback",
            "human_modifications": modifications,
            "rollback_target": resolved_target,
            "current_image_path": rollback_uri,
            "asset_graph": prune_asset_graph(state.get("asset_graph", []), rollback_uri),
            "sub_goals": [],
            "current_sub_goal_idx": 0,
            "review_request": review_payload,
        }

    if decision == "modify":
        updated = _apply_human_modifications(state, modifications)
        if updated["changed"]:
            return {
                "human_decision": "modify",
                "human_modifications": modifications,
                "sub_goals": updated["sub_goals"],
                "current_sub_goal_idx": updated["start_idx"],
                "current_image_path": updated["resume_image"],
                "asset_graph": updated["asset_graph"],
                "review_request": review_payload,
            }
        # Empty/invalid modify payload falls back to approve.
        decision = "approve"

    return {
        "human_decision": decision,
        "human_modifications": modifications,
        "rollback_target": rollback_target if isinstance(rollback_target, dict) else {},
        "review_request": review_payload,
    }


# ---------------------------------------------------------------------------
# 5. Context Fold Node
# ---------------------------------------------------------------------------

def context_fold_node(state: AgentState) -> dict:
    """
    Fold context after a completed turn:
    - Append ActionRecord to action_history
    - Clear tool_contexts
    - Advance turn_id
    - Reset transient state

    Writes: action_history, tool_contexts, turn_id, human_decision, sub_goals,
            quality_assessment, error_message
    """
    from .state import ActionRecord
    from dataclasses import asdict

    # Build action record for this turn
    plan_summary = "; ".join(
        sg.get("description", "") for sg in state.get("sub_goals", [])
    )

    # Determine outcome
    statuses = [sg.get("status", "pending") for sg in state.get("sub_goals", [])]
    if state.get("human_decision") == "rollback":
        outcome = "rolled_back"
    elif all(s == "completed" for s in statuses):
        outcome = "completed"
    else:
        outcome = "partial"

    record = ActionRecord(
        turn_id=state.get("turn_id", 0),
        intent=state.get("user_message", ""),
        plan_summary=plan_summary,
        outcome=outcome,
        validated_asset_uri=state.get("current_image_path", ""),
    )

    current_image = state.get("current_image_path", "")
    pruned_assets = prune_asset_graph(state.get("asset_graph", []), current_image)

    print(f"[ContextFold] Turn {state.get('turn_id', 0)} → {outcome}")

    return {
        "action_history": [asdict(record)],  # append via reducer
        "asset_graph": pruned_assets,
        "tool_contexts": [],                  # clear ephemeral data
        "turn_id": state.get("turn_id", 0) + 1,
        "replan_attempts": 0,
        "human_decision": "",
        "human_modifications": {},
        "review_request": {},
        "rollback_target": {},
        "sub_goals": [],
        "current_sub_goal_idx": 0,
        "quality_assessment": "",
        "error_message": "",
    }


# ---------------------------------------------------------------------------
# 6. Error Handler Node
# ---------------------------------------------------------------------------

def error_handler_node(state: AgentState) -> dict:
    """
    Handle errors from planner/executor.
    Log the error and prepare state for retry via planner.

    Writes: error_message, sub_goals
    """
    error = state.get("error_message", "Unknown error")
    print(f"[ErrorHandler] {error}")

    # Increment retry counts on failed sub-goals
    sub_goals = [dict(sg) for sg in state.get("sub_goals", [])]
    for sg in sub_goals:
        if sg.get("status") == "failed":
            sg["retry_count"] = sg.get("retry_count", 0) + 1
            sg["status"] = "pending"

    return {
        "sub_goals": sub_goals,
        "error_message": "",
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_planner(state: AgentState) -> str:
    """Route after planner: executor | context_fold | error_handler."""
    if state.get("error_message"):
        return "error_handler"
    if not state.get("sub_goals") or state.get("is_complete"):
        return "context_fold"
    return "executor"


def route_after_quality(state: AgentState) -> str:
    """Route after quality: context_fold (auto-approve) | human_review | planner."""
    score = state.get("quality_score", 0.0)
    min_pass_score = float(_cfg(state, "quality", "min_pass_score", default=0.6))
    auto_threshold = float(_cfg(state, "quality", "auto_approve_threshold", default=0.85))

    # Check retry budgets for both failed sub-goals and quality-driven replans.
    max_retries = int(_cfg(state, "planner", "max_retries_per_sub_goal", default=2))
    any_over_retry = any(
        sg.get("retry_count", 0) >= max_retries
        for sg in state.get("sub_goals", [])
    )
    replan_attempts = int(state.get("replan_attempts", 0))
    quality_retry_exhausted = replan_attempts >= max_retries

    # Auto-approve good results (skip interrupt/human review).
    if score >= auto_threshold and state.get("quality_pass", score >= min_pass_score):
        return "context_fold"

    if score < min_pass_score and not any_over_retry and not quality_retry_exhausted:
        # Poor quality, retry allowed
        return "planner"

    # Default: human decides
    return "human_review"


def route_after_human(state: AgentState) -> str:
    """Route after human review: context_fold | planner | executor."""
    decision = state.get("human_decision", "approve")

    if decision == "approve":
        return "context_fold"
    if decision in ("rollback", "replan"):
        return "planner"
    if decision == "modify":
        return "executor"

    return "context_fold"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fallback_parse_planner_response(engine, response: str) -> list[dict]:
    """
    Compatibility parser for models that output a plain adjustments dict instead of
    the required sub-goal array.
    """
    parsed = engine.extract_json_from_response(response)
    if (not isinstance(parsed, dict)) or (not parsed):
        adjustments = _extract_adjustments_from_text(response)
        if adjustments:
            return [
                {
                    "id": 1,
                    "stage_type": "global",
                    "operation_category": _infer_operation_category_from_adjustments(adjustments),
                    "description": "Apply requested global adjustments",
                    "adjustments": adjustments,
                    "local_specs": [],
                }
            ]
        return []

    if isinstance(parsed.get("sub_goals"), list):
        return parsed["sub_goals"]

    if isinstance(parsed.get("plan"), list):
        return parsed["plan"]

    # Treat as one global adjustment stage (legacy MonetGPT-style response).
    adjustments = {}
    for key, value in parsed.items():
        if isinstance(value, (int, float)):
            adjustments[key] = value

    if not adjustments:
        return []

    return [
        {
            "id": 1,
            "stage_type": "global",
            "operation_category": _infer_operation_category_from_adjustments(adjustments),
            "description": "Apply requested global adjustments",
            "adjustments": adjustments,
            "local_specs": [],
        }
    ]


def _extract_adjustments_from_text(response: str) -> dict:
    """Best-effort extraction for non-JSON planner responses."""
    if not isinstance(response, str):
        return {}

    ops = [
        "Exposure",
        "Contrast",
        "Highlights",
        "Shadows",
        "Whites",
        "Blacks",
        "Temperature",
        "Tint",
        "Saturation",
        "Vibrance",
        "Dehaze",
    ]

    text = response.replace("\n", " ")
    adjustments = {}

    # Pattern A: "Exposure: +8" or "Exposure = -12"
    for op in ops:
        m = re.search(
            rf"\b{re.escape(op)}\b\s*[:=]\s*([+-]?\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            adjustments[op] = int(round(float(m.group(1))))

    # Pattern B: "value +8 for Exposure"
    for op in ops:
        m = re.search(
            rf"value\s*([+-]?\d+(?:\.\d+)?)\s*(?:for|to)\s*{re.escape(op)}",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            adjustments[op] = int(round(float(m.group(1))))

    return adjustments


def _infer_operation_category_from_adjustments(adjustments: dict) -> str:
    """Infer planner operation category from adjustment keys."""
    keys = set(adjustments.keys())

    if any(k in {"Temperature", "Tint", "Saturation"} for k in keys):
        return "color-temperature"

    if any(
        k.startswith("HueAdjustment")
        or k.startswith("SaturationAdjustment")
        or k.startswith("LuminanceAdjustment")
        for k in keys
    ):
        return "hsl"

    return "white-balance-tone-contrast"


def _operation_category_to_stage(category: str) -> int | None:
    """Map operation_category string to stage index for get_processed_predictions."""
    mapping = {
        "white-balance-tone-contrast": 0,
        "color-temperature": 1,
        "hsl": 2,
    }
    return mapping.get(category)


def _cfg(state: AgentState, *keys: str, default: Any = None) -> Any:
    """Read nested config from state['agent_config'] with safe defaults."""
    value: Any = state.get("agent_config", {})
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
        if value is None:
            return default
    return value


def _next_plan_version(state: AgentState) -> int:
    """Increment version only on replans/quality-driven replans; keep version on first plan."""
    current = int(state.get("plan_version", 0))
    if state.get("human_decision") in ("replan", "rollback"):
        return current + 1
    if state.get("quality_assessment") and not state.get("quality_pass", True):
        return current + 1
    if not state.get("sub_goals"):
        return current
    return current + 1


def _next_replan_attempts(state: AgentState) -> int:
    """
    Track quality-driven/human-triggered replans within the current turn.

    Reset happens in context_fold_node once a turn is finalized.
    """
    current = int(state.get("replan_attempts", 0))
    if state.get("human_decision") == "replan":
        return current + 1
    if state.get("quality_assessment") and not state.get("quality_pass", True):
        return current + 1
    return current


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _upsert_asset_node(asset_graph: list[dict], node: dict) -> list[dict]:
    """Replace existing asset node by URI, or append if it does not exist."""
    uri = node.get("uri")
    if not uri:
        return asset_graph
    updated = []
    replaced = False
    for item in asset_graph:
        if item.get("uri") == uri:
            updated.append(node)
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        updated.append(node)
    return updated


def _resolve_rollback_uri(state: AgentState, rollback_target: dict) -> tuple[str, dict]:
    """
    Resolve rollback target to a valid URI.

    Priority:
      1) explicit asset_uri
      2) turn_id (+ optional step_id)
      3) parent of current image (single-step undo)
      4) original image
    """
    asset_graph = [dict(node) for node in state.get("asset_graph", [])]
    by_uri = {n.get("uri"): n for n in asset_graph if n.get("uri")}
    current_uri = state.get("current_image_path", "")
    original_uri = state.get("original_image_path", "")

    target = rollback_target if isinstance(rollback_target, dict) else {}
    target_uri = target.get("asset_uri")
    if target_uri in by_uri or target_uri == original_uri:
        return target_uri, {"asset_uri": target_uri}

    turn_id = target.get("turn_id")
    step_id = target.get("step_id")
    if turn_id is not None:
        candidates = [n for n in asset_graph if n.get("turn_id") == turn_id]
        if step_id is not None:
            exact = [n for n in candidates if n.get("step_id") == step_id]
            if exact:
                uri = exact[-1].get("uri", original_uri)
                return uri, {"turn_id": turn_id, "step_id": step_id, "asset_uri": uri}
        if candidates:
            candidates.sort(key=lambda n: int(n.get("step_id", -1)))
            uri = candidates[-1].get("uri", original_uri)
            return uri, {"turn_id": turn_id, "asset_uri": uri}

    current_node = by_uri.get(current_uri, {})
    parent_uri = current_node.get("parent_uri")
    if parent_uri:
        return parent_uri, {"asset_uri": parent_uri}

    return original_uri, {"asset_uri": original_uri}


def _apply_human_modifications(state: AgentState, modifications: dict) -> dict:
    """
    Merge human modifications into sub-goals and prepare executor resume point.

    Accepted payloads:
      - {"sub_goal_updates": [{"id": 2, "adjustments": {...}}, ...]}
      - {"sub_goal_id": 2, "adjustments": {...}}
      - {"overrides": {"2": {"adjustments": {...}}, "3": {...}}}
    """
    sub_goals = [dict(sg) for sg in state.get("sub_goals", [])]
    if not isinstance(modifications, dict) or not sub_goals:
        return {
            "changed": False,
            "sub_goals": sub_goals,
            "start_idx": state.get("current_sub_goal_idx", 0),
            "resume_image": state.get("current_image_path", ""),
            "asset_graph": state.get("asset_graph", []),
        }

    updates: list[dict] = []
    if isinstance(modifications.get("sub_goal_updates"), list):
        updates = [u for u in modifications.get("sub_goal_updates", []) if isinstance(u, dict)]
    elif isinstance(modifications.get("overrides"), dict):
        for key, payload in modifications["overrides"].items():
            if isinstance(payload, dict):
                try:
                    goal_id = int(key)
                except (TypeError, ValueError):
                    continue
                updates.append({"id": goal_id, **payload})
    elif "sub_goal_id" in modifications:
        payload = {k: v for k, v in modifications.items() if k != "sub_goal_id"}
        updates.append({"id": modifications["sub_goal_id"], **payload})

    changed_indices: list[int] = []
    mutable_fields = {"stage_type", "operation_category", "description", "adjustments", "local_specs"}
    for upd in updates:
        goal_id = upd.get("id")
        for idx, sg in enumerate(sub_goals):
            if sg.get("id") != goal_id:
                continue
            for field in mutable_fields:
                if field in upd:
                    sg[field] = upd[field]
                    changed_indices.append(idx)
            break

    if not changed_indices:
        return {
            "changed": False,
            "sub_goals": sub_goals,
            "start_idx": state.get("current_sub_goal_idx", 0),
            "resume_image": state.get("current_image_path", ""),
            "asset_graph": state.get("asset_graph", []),
        }

    start_idx = min(changed_indices)
    for idx in range(start_idx, len(sub_goals)):
        sub_goals[idx]["status"] = "pending"
        sub_goals[idx]["retry_count"] = 0

    resume_image = _find_input_uri_for_step(
        state=state,
        step_idx=start_idx,
    )

    # Remove stale assets from this turn at/after start_idx to keep gallery and rollback clean.
    turn_id = state.get("turn_id", 0)
    pruned_assets = []
    for node in state.get("asset_graph", []):
        node_turn = node.get("turn_id")
        node_step = int(node.get("step_id", -1))
        if node_turn == turn_id and node_step >= start_idx:
            continue
        pruned_assets.append(node)

    return {
        "changed": True,
        "sub_goals": sub_goals,
        "start_idx": start_idx,
        "resume_image": resume_image,
        "asset_graph": pruned_assets,
    }


def _find_input_uri_for_step(state: AgentState, step_idx: int) -> str:
    """Find image URI that should be used as input before executing step_idx."""
    asset_graph = state.get("asset_graph", [])
    turn_id = state.get("turn_id", 0)
    original = state.get("original_image_path", "")

    if step_idx <= 0:
        # Start of turn: use the image before first step. If we can find step0 asset, use its parent.
        for node in asset_graph:
            if node.get("turn_id") == turn_id and int(node.get("step_id", -1)) == 0:
                return node.get("parent_uri") or original
        return original

    # Prefer exact step node parent.
    for node in asset_graph:
        if node.get("turn_id") == turn_id and int(node.get("step_id", -1)) == step_idx:
            return node.get("parent_uri") or original

    # Fallback to output of previous step if available.
    prev = None
    for node in asset_graph:
        if node.get("turn_id") == turn_id and int(node.get("step_id", -1)) == step_idx - 1:
            prev = node
            break
    if prev:
        return prev.get("uri") or original

    return state.get("current_image_path", original)
