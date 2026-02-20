"""
Multi-turn reasoning generator using VLM.

For each trajectory from generate_puzzle_mt.py, sends the multi-turn
conversation (with all intermediate images) to the VLM and generates
context-aware reasoning for each turn.

Usage:
    python dataset/query_llm_multiturn.py \\
        --trajectories data/multiturn_trajectories/all_trajectories.json \\
        --output data/multiturn_reasoning.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Dict, Any

from inference.core import InferenceEngine


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional photo-editing reasoning engine.

You will be given a multi-turn editing conversation where each turn shows:
- The user's editing request
- The current state of the image

For each turn, provide a detailed analysis of:
1. What the image currently looks like
2. What the user wants to change
3. Why specific adjustments are appropriate
4. How this turn relates to previous editing decisions

Be specific about colors, tones, regions, and technical details."""


def _create_turn_prompt(turn: dict, turn_idx: int, prev_turns: list) -> str:
    """Create the reasoning prompt for a single turn within a trajectory."""
    context = ""
    if prev_turns:
        context = "## Previous turns:\n"
        for pt in prev_turns:
            context += f"- Turn {pt['turn_id']}: {pt['description']}\n"
        context += "\n"

    return (
        f"{context}"
        f"## Current Turn {turn_idx}\n"
        f"Turn type: {turn.get('turn_type', 'unknown')}\n"
        f"Description: {turn.get('description', '')}\n"
        f"Config: {json.dumps(turn.get('config', {}))}\n\n"
        f"Analyze this editing step. Explain the reasoning behind each adjustment, "
        f"considering the image content and any previous edits. "
        f"Be specific about what visual issues are being addressed."
    )


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_reasoning(
    trajectories_path: str,
    output_path: str,
    inference_config: str = "configs/inference_config.yaml",
    timeout: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate multi-turn reasoning for all trajectories.

    Args:
        trajectories_path: Path to all_trajectories.json.
        output_path: Path to save reasoning output.
        inference_config: Inference config path.
        timeout: Seconds between API calls.

    Returns:
        List of trajectories with added reasoning.
    """
    engine = InferenceEngine(inference_config)

    with open(trajectories_path, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    results = []

    for traj_idx, trajectory in enumerate(trajectories):
        print(f"[{traj_idx + 1}/{len(trajectories)}] Processing trajectory...")

        turns = trajectory.get("turns", [])
        reasoned_turns = []

        for turn_idx, turn in enumerate(turns):
            image_path = turn.get("output_image", turn.get("input_image", ""))

            if not os.path.exists(image_path):
                # Use input image if output doesn't exist
                image_path = turn.get("input_image", "")

            if not image_path or not os.path.exists(image_path):
                print(f"  Warning: No image for turn {turn_idx}, skipping reasoning.")
                turn["reasoning"] = ""
                reasoned_turns.append(turn)
                continue

            # Generate reasoning
            turn_prompt = _create_turn_prompt(turn, turn_idx, reasoned_turns)

            try:
                reasoning = engine.query_structured(
                    image_path=image_path,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=turn_prompt,
                    temperature=0.3,
                )
                turn["reasoning"] = reasoning
                print(f"  Turn {turn_idx}: Generated {len(reasoning)} chars of reasoning")
            except Exception as e:
                print(f"  Turn {turn_idx}: Error generating reasoning: {e}")
                turn["reasoning"] = ""

            reasoned_turns.append(turn)

            if timeout > 0:
                time.sleep(timeout)

        trajectory["turns"] = reasoned_turns
        results.append(trajectory)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated reasoning for {len(results)} trajectories → {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn VLM reasoning")
    parser.add_argument("--trajectories", required=True, help="Path to all_trajectories.json")
    parser.add_argument("--output", default="data/multiturn_reasoning.json")
    parser.add_argument("--config", default="configs/inference_config.yaml")
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    generate_reasoning(args.trajectories, args.output, args.config, args.timeout)


if __name__ == "__main__":
    main()
