"""
Multi-turn ShareGPT format dataset creator.

Converts multi-turn trajectories (with reasoning from query_llm_multiturn.py)
into the extended ShareGPT conversation format:

    system → user_turn1 → assistant_analysis1 → user_json_prompt1 → assistant_json1 →
    user_turn2 → assistant_analysis2 → user_json_prompt2 → assistant_json2 → ...

Usage:
    python dataset/create_datasets_multiturn.py \\
        --input data/multiturn_reasoning.json \\
        --output data/sharegpt_puzzle_mt.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any

from dataset.constants import OPERATION_MAP, SUMMARY_MAP


# ---------------------------------------------------------------------------
# ShareGPT conversation builder
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a helpful advanced image-editing assistant with expertise in Adobe Lightroom "
    "and multi-turn interactive photo editing. You can handle iterative editing requests, "
    "refine previous adjustments, perform local/regional edits, and roll back changes "
    "when needed. You maintain context across multiple editing turns."
)


def _turn_type_to_operation_category(turn_type: str, config: Any) -> str:
    """Map turn type to the appropriate operation category."""
    if turn_type == "local_edit" or isinstance(config, list):
        return "local-editing"
    # For global turns, infer from the ops present
    if isinstance(config, dict):
        from dataset.constants import WHITE_BALANCE_TONE_CONTRAST, COLOR_TEMPERATURE
        ops = set(config.keys())
        if ops & set(WHITE_BALANCE_TONE_CONTRAST):
            return "white-balance-tone-contrast"
        if ops & set(COLOR_TEMPERATURE):
            return "color-temperature"
        if any("Adjustment" in k for k in ops):
            return "hsl"
    return "white-balance-tone-contrast"


def _create_analysis_prompt(turn: dict, turn_idx: int, total_turns: int) -> str:
    """Create the user prompt asking for analysis of this turn."""
    turn_type = turn.get("turn_type", "global_init")
    description = turn.get("description", "")
    op_category = _turn_type_to_operation_category(turn_type, turn.get("config"))
    op_desc = OPERATION_MAP.get(op_category, "")

    context = ""
    if turn_idx > 0:
        context = (
            f"This is turn {turn_idx + 1} of {total_turns} in an interactive editing session. "
            f"The image has already been partially edited in previous turns. "
        )

    return (
        f"{context}"
        f"Analyze the provided image and develop a professional-grade editing plan "
        f"using operations available in Adobe Lightroom to address issues in {op_desc}.\n\n"
        f"User request: {description}\n\n"
        f"Create a professional editing plan for this turn."
    )


def _create_json_prompt(turn: dict) -> str:
    """Create the user prompt asking for JSON output of this turn."""
    turn_type = turn.get("turn_type", "global_init")
    op_category = _turn_type_to_operation_category(turn_type, turn.get("config"))
    summary = SUMMARY_MAP.get(op_category, "")

    if isinstance(turn.get("config"), list):
        return (
            "Based on your regional editing analysis, output the optimal local adjustments "
            "as a JSON array. Each element represents one targeted regional edit.\n\n"
            f"{summary}"
        )
    return (
        "Based on the editing plan, tell the optimal adjustment values needed "
        "to edit this photo in JSON format. All adjustment values are scaled between "
        "-100 and +100.\n\n"
        f"{summary}"
    )


def trajectory_to_sharegpt(trajectory: dict) -> Dict[str, Any]:
    """
    Convert a multi-turn trajectory into ShareGPT conversation format.

    Returns:
        ShareGPT entry dict with "conversations" key.
    """
    conversations = [
        {"from": "system", "value": SYSTEM_MESSAGE},
    ]

    turns = trajectory.get("turns", [])
    total_turns = len(turns)

    for turn_idx, turn in enumerate(turns):
        # User: analysis prompt
        analysis_prompt = _create_analysis_prompt(turn, turn_idx, total_turns)
        conversations.append({"from": "human", "value": analysis_prompt})

        # Assistant: reasoning/analysis
        reasoning = turn.get("reasoning", "")
        if not reasoning:
            reasoning = f"Analyzing turn {turn_idx + 1}: {turn.get('description', '')}."
        conversations.append({"from": "gpt", "value": reasoning})

        # User: JSON prompt
        json_prompt = _create_json_prompt(turn)
        conversations.append({"from": "human", "value": json_prompt})

        # Assistant: JSON output
        config = turn.get("config", {})
        if isinstance(config, list):
            json_answer = json.dumps(config, indent=2, ensure_ascii=False)
        else:
            json_answer = json.dumps(config, indent=2, ensure_ascii=False)
        conversations.append({"from": "gpt", "value": json_answer})

    return {
        "conversations": conversations,
        "source_image": trajectory.get("source_image", ""),
        "style": trajectory.get("style", "balanced"),
        "num_turns": total_turns,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create multi-turn ShareGPT dataset")
    parser.add_argument("--input", required=True, help="Path to multiturn_reasoning.json")
    parser.add_argument("--output", default="data/sharegpt_puzzle_mt.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    sharegpt_entries = []
    for traj in trajectories:
        entry = trajectory_to_sharegpt(traj)
        sharegpt_entries.append(entry)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sharegpt_entries, f, indent=2, ensure_ascii=False)

    print(f"Created {len(sharegpt_entries)} multi-turn ShareGPT entries → {args.output}")


if __name__ == "__main__":
    main()
