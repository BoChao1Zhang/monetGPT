"""
Multi-turn trajectory generator for puzzle_mt dataset.

For each source image, generates 3-6 turn editing trajectories:
  Turn 1: global_init   — initial global adjustments (exposure, WB, contrast)
  Turn 2: global_refine  — refine/correct (fix overexposure, clipped highlights, etc.)
  Turn 3: local_edit     — local/regional edits (skin, sky, etc.)
  Turn 4: rollback_correct — undo previous + apply alternative approach
  Turn 5: style_shift    — switch style (balanced → vibrant, etc.)

Each turn produces:
  - Config (JSON dict or array)
  - Executed intermediate image
  - Turn metadata

Output: Complete trajectory JSON per source image.

Usage:
    python dataset/generate_puzzle_mt.py \\
        --source-dir data/source_images \\
        --output-dir data/multiturn_trajectories \\
        --num-turns 5 \\
        --num-trajectories 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from typing import List, Dict, Any

from dataset.constants import (
    WHITE_BALANCE_TONE_CONTRAST,
    COLOR_TEMPERATURE,
    COLOR_SPECIFIC_ADJUSTMENTS,
    MULTI_TURN_TYPES,
)


# ---------------------------------------------------------------------------
# Turn templates — each returns (turn_type, config, description)
# ---------------------------------------------------------------------------

def _sample_global_init() -> tuple[str, dict, str]:
    """Sample initial global adjustments (Turn 1: exposure, WB, contrast)."""
    ops = {}
    # Sample 2-4 operations from white-balance-tone-contrast
    n = random.randint(2, 4)
    chosen = random.sample(WHITE_BALANCE_TONE_CONTRAST, min(n, len(WHITE_BALANCE_TONE_CONTRAST)))
    for op in chosen:
        ops[op] = random.randint(-60, 60)

    desc = "Apply initial global adjustments: " + ", ".join(
        f"{k}{'+' if v > 0 else ''}{v}" for k, v in ops.items()
    )
    return "global_init", ops, desc


def _sample_global_refine(prev_config: dict) -> tuple[str, dict, str]:
    """Sample refinement adjustments (Turn 2: correct overexposure etc.)."""
    ops = {}
    # Pick 1-3 ops, apply corrective values (opposite direction or small adjustments)
    candidates = list(WHITE_BALANCE_TONE_CONTRAST) + list(COLOR_TEMPERATURE)
    n = random.randint(1, 3)
    chosen = random.sample(candidates, min(n, len(candidates)))
    for op in chosen:
        prev_val = prev_config.get(op, 0)
        if prev_val != 0:
            # Correct by reducing the previous value
            correction = -int(prev_val * random.uniform(0.3, 0.7))
            ops[op] = correction
        else:
            ops[op] = random.randint(-30, 30)

    desc = "Refine global adjustments: " + ", ".join(
        f"{k}{'+' if v > 0 else ''}{v}" for k, v in ops.items()
    )
    return "global_refine", ops, desc


def _sample_local_edit() -> tuple[str, list, str]:
    """Sample local/regional edits (Turn 3)."""
    regions = [
        ("skin", "human skin area"),
        ("sky", "sky"),
        ("background", "background"),
        ("vegetation", "trees and grass"),
        ("hair", "hair"),
    ]
    local_ops = ["brightness", "contrast", "saturation", "temperature", "vibrance"]

    n = random.randint(2, 4)
    chosen_regions = random.sample(regions, min(n, len(regions)))

    specs = []
    for region_name, mask_prompt in chosen_regions:
        op = random.choice(local_ops)
        value = round(random.uniform(-0.5, 0.5), 2)
        if value == 0:
            value = 0.1
        specs.append({
            "region": region_name,
            "mask_type": "semantic",
            "mask_prompt": mask_prompt,
            "op": op,
            "value": value,
            "refine": {"feather": random.choice([3, 5, 8])},
            "reason_issue": f"The {region_name} region needs {op} adjustment.",
            "reason_solution": f"Apply {op} {'+' if value > 0 else ''}{value} to {region_name}.",
        })

    desc = f"Apply {len(specs)} local edits: " + ", ".join(
        f"{s['region']}/{s['op']}" for s in specs
    )
    return "local_edit", specs, desc


def _sample_rollback_correct(turn_history: list) -> tuple[str, dict, str]:
    """Sample rollback + alternative (Turn 4)."""
    ops = {}
    # Sample new global adjustments as the 'alternative'
    candidates = list(WHITE_BALANCE_TONE_CONTRAST) + list(COLOR_TEMPERATURE)
    n = random.randint(2, 3)
    chosen = random.sample(candidates, min(n, len(candidates)))
    for op in chosen:
        ops[op] = random.randint(-40, 40)

    rollback_turn = max(0, len(turn_history) - 2) if turn_history else 0
    desc = f"Rollback to turn {rollback_turn} and apply alternative: " + ", ".join(
        f"{k}{'+' if v > 0 else ''}{v}" for k, v in ops.items()
    )
    return "rollback_correct", ops, desc


def _sample_style_shift(current_style: str) -> tuple[str, dict, str]:
    """Sample style shift (Turn 5)."""
    styles = ["balanced", "vibrant", "retro"]
    new_style = random.choice([s for s in styles if s != current_style])

    # Style shifts typically involve saturation + temperature + contrast
    ops = {}
    if new_style == "vibrant":
        ops = {"Saturation": random.randint(15, 40), "Contrast": random.randint(10, 25)}
    elif new_style == "retro":
        ops = {"Saturation": random.randint(-30, -10), "Temperature": random.randint(5, 20)}
    else:
        ops = {"Saturation": random.randint(-10, 10), "Contrast": random.randint(-10, 10)}

    desc = f"Shift style from {current_style} to {new_style}: " + ", ".join(
        f"{k}{'+' if v > 0 else ''}{v}" for k, v in ops.items()
    )
    return "style_shift", ops, desc


# ---------------------------------------------------------------------------
# Trajectory generator
# ---------------------------------------------------------------------------

TURN_GENERATORS = {
    "global_init": lambda hist, cfg, style: _sample_global_init(),
    "global_refine": lambda hist, cfg, style: _sample_global_refine(cfg),
    "local_edit": lambda hist, cfg, style: _sample_local_edit(),
    "rollback_correct": lambda hist, cfg, style: _sample_rollback_correct(hist),
    "style_shift": lambda hist, cfg, style: _sample_style_shift(style),
}


def generate_trajectory(
    source_image: str,
    output_dir: str,
    num_turns: int = 5,
    style: str = "balanced",
    execute: bool = False,
) -> Dict[str, Any]:
    """
    Generate a multi-turn editing trajectory for one source image.

    Args:
        source_image: Path to source image.
        output_dir: Directory to save intermediate images.
        num_turns: Number of editing turns (3-6).
        style: Initial editing style.
        execute: If True, actually execute edits via pipeline (requires GPU).

    Returns:
        Trajectory dict with turns, configs, descriptions.
    """
    num_turns = min(max(num_turns, 3), 6)
    os.makedirs(output_dir, exist_ok=True)

    # Copy original
    ext = os.path.splitext(source_image)[1]
    original_copy = os.path.join(output_dir, f"original{ext}")
    shutil.copy2(source_image, original_copy)

    trajectory = {
        "source_image": source_image,
        "style": style,
        "turns": [],
    }

    # Determine turn type sequence
    turn_sequence = _sample_turn_sequence(num_turns)

    turn_history = []
    accumulated_config = {}
    current_image = original_copy

    for turn_idx, turn_type in enumerate(turn_sequence):
        generator = TURN_GENERATORS[turn_type]
        turn_type_out, config, description = generator(
            turn_history, accumulated_config, style
        )

        output_path = os.path.join(output_dir, f"turn{turn_idx}{ext}")

        turn_data = {
            "turn_id": turn_idx,
            "turn_type": turn_type_out,
            "description": description,
            "config": config,
            "input_image": current_image,
            "output_image": output_path,
        }

        if execute:
            try:
                from pipeline.core import ImageEditingPipeline
                pipeline = ImageEditingPipeline()
                is_local = isinstance(config, list)
                pipeline.execute_single_stage(config, current_image, output_path, is_local=is_local)
                turn_data["executed"] = True
                current_image = output_path
            except Exception as e:
                turn_data["executed"] = False
                turn_data["error"] = str(e)
        else:
            turn_data["executed"] = False

        # Update accumulated state
        if isinstance(config, dict):
            accumulated_config.update(config)
        if turn_type == "rollback_correct" and len(turn_history) >= 2:
            rollback_target = turn_history[-2].get("output_image", original_copy)
            current_image = rollback_target
            turn_data["rollback_to"] = rollback_target

        turn_history.append(turn_data)
        trajectory["turns"].append(turn_data)

    return trajectory


def _sample_turn_sequence(num_turns: int) -> List[str]:
    """Sample a logical sequence of turn types."""
    # Always start with global_init
    sequence = ["global_init"]

    # Remaining turns are sampled with logical constraints
    pool = ["global_refine", "local_edit", "rollback_correct", "style_shift"]

    for i in range(1, num_turns):
        if i == 1:
            # Turn 2 is usually a refinement
            choice = random.choice(["global_refine", "local_edit"])
        elif i == num_turns - 1 and random.random() < 0.3:
            # Last turn sometimes a style shift
            choice = "style_shift"
        else:
            choice = random.choice(pool)
        sequence.append(choice)

    return sequence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn editing trajectories")
    parser.add_argument("--source-dir", required=True, help="Directory of source images")
    parser.add_argument("--output-dir", default="data/multiturn_trajectories")
    parser.add_argument("--num-turns", type=int, default=5)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--execute", action="store_true", help="Execute edits via pipeline")
    parser.add_argument("--style", default="balanced")
    args = parser.parse_args()

    # Collect source images
    valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    source_images = sorted([
        os.path.join(args.source_dir, f)
        for f in os.listdir(args.source_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if not source_images:
        print(f"No images found in {args.source_dir}")
        return

    print(f"Found {len(source_images)} source images")
    all_trajectories = []

    for i in range(min(args.num_trajectories, len(source_images))):
        src = source_images[i % len(source_images)]
        img_name = os.path.splitext(os.path.basename(src))[0]
        traj_dir = os.path.join(args.output_dir, f"traj_{i:04d}_{img_name}")

        print(f"[{i+1}/{args.num_trajectories}] Generating trajectory for {src}...")

        num_turns = random.randint(3, min(args.num_turns, 6))
        trajectory = generate_trajectory(
            src, traj_dir, num_turns=num_turns, style=args.style, execute=args.execute,
        )
        all_trajectories.append(trajectory)

        # Save individual trajectory
        traj_path = os.path.join(traj_dir, "trajectory.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, indent=2, ensure_ascii=False)

    # Save all trajectories
    output_path = os.path.join(args.output_dir, "all_trajectories.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(all_trajectories)} trajectories → {output_path}")


if __name__ == "__main__":
    main()
