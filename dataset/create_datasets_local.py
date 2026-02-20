"""
Dataset creation for Puzzle C (local/regional editing) in ShareGPT format.

Two-round conversation structure (similar to Puzzle 3):
  Round 1: Image + local analysis prompt → regional reasoning
  Round 2: JSON generation prompt → JSON array of local edit specs
"""
import json
import glob
import os
from .utils import load_config


def create_sharegpt_format_puzzle_c(plan, reasoning, image_path):
    """
    Create ShareGPT format entry for Puzzle C.

    Args:
        plan: List of local edit spec dicts (the JSON array ground truth).
        reasoning: VLM-generated reasoning text.
        image_path: Path to the source image.

    Returns:
        ShareGPT-format dict with 5-turn conversation.
    """
    plan_json = f"```json\n{json.dumps(plan, indent=2)}\n```"

    analysis_prompt = """
<image>

Analyze the provided image and develop a professional-grade **local/regional editing plan**. Focus on region-specific issues that cannot be addressed by global adjustments alone.

Examine each semantic region of the image (such as skin, sky, hair, background, vegetation, clothing, shadows, highlights) and identify localized issues that need targeted correction.

For each regional issue found, provide:

**Region:** [Name the region, e.g., "skin", "sky", "background", "hair", "vegetation"]
**Issue:** [Describe the specific localized problem — e.g., "The skin tones appear overly warm and slightly oversaturated, making the subject look unnatural"]
**Solution:** [Describe the targeted adjustment — e.g., "Reduce temperature and slightly decrease saturation specifically on the skin area"]

Guidelines:
- Focus on 2-5 of the most impactful regional adjustments.
- Only suggest local edits where a global adjustment would negatively affect other regions.
- Each adjustment should target ONE region with ONE operation for precise control.
"""

    json_prompt = """
Based on your regional editing analysis and the original image, output the **optimal** local adjustments as a JSON array. Each element represents one targeted regional edit.

Each element must contain: region, mask_type ("semantic" or "luminance_range"), mask_prompt, op (brightness/contrast/saturation/temperature/tint/vibrance/highlights/shadows/whites/blacks/dehaze/exposure), value (float in [-1.0, +1.0]), refine (dict with optional feather/invert/luminance_range), reason_issue, reason_solution.

Output ONLY the JSON array with 2-5 elements.
"""

    answer_2 = f"""
Applying the below local/regional adjustments will make the image **optimal**.

{plan_json}
"""

    entry = {
        "messages": [
            {
                "content": "You are a helpful advanced image-editing assistant with expertise in Adobe Lightroom and local/regional photo editing.",
                "role": "system",
            },
            {
                "content": analysis_prompt,
                "role": "user",
            },
            {
                "content": reasoning,
                "role": "assistant",
            },
            {
                "content": json_prompt,
                "role": "user",
            },
            {
                "content": answer_2,
                "role": "assistant",
            },
        ]
    }
    if image_path:
        entry["images"] = [image_path]
    return entry


def create_dataset_puzzle_c(config_path="configs/dataset_config.yaml"):
    """Create dataset for Puzzle C."""
    config = load_config(config_path)
    puzzle_config = config["puzzles"]["puzzle_c"]
    generation_config = config["generation"]

    messages = []
    reasoning_files_path = puzzle_config["reasoning_path"]
    images_base_path = puzzle_config["images_base_path"]
    configs_base_path = generation_config["output_dirs"]["puzzle_c"]

    train_files = glob.glob(reasoning_files_path)

    # Get image extension from config pattern
    images_path_pattern = puzzle_config["images_path"]
    image_ext = images_path_pattern.split("*")[-1]

    data_entry_count = 0

    for reasoning_file_path in train_files:
        filename = os.path.splitext(os.path.basename(reasoning_file_path))[0]

        # Load reasoning
        with open(reasoning_file_path, "r", encoding="utf-8") as f:
            reasoning = f.read()

        if reasoning == "TypeError" or len(reasoning) < 3:
            print(f"Skipping {filename}: invalid reasoning")
            continue

        # Load local edit plan
        config_file = os.path.join(configs_base_path, f"{filename}.json")
        if not os.path.exists(config_file):
            print(f"Config not found: {config_file}")
            continue

        with open(config_file, "r") as f:
            plan = json.load(f)

        # Find source image
        image_path = os.path.join(images_base_path, f"{filename}{image_ext}")
        if not os.path.exists(image_path):
            # Try without _local suffix
            base_name = filename.replace("_local", "")
            image_path = os.path.join(images_base_path, f"{base_name}{image_ext}")

        if not os.path.exists(image_path):
            print(f"Image not found for {filename}")
            continue

        data_entry = create_sharegpt_format_puzzle_c(plan, reasoning, image_path)
        messages.append(data_entry)
        data_entry_count += 1

    output_file = puzzle_config["output_file"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(messages, indent=4, ensure_ascii=False))
        f.write("\n")

    print(f"Puzzle C data written to {output_file}")
    print(f"data_entry_count: {data_entry_count}")
    return messages
