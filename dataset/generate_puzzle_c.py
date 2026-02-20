"""
Puzzle C configuration generator: local/regional editing data synthesis.

For each source image:
1. Detect semantic regions via GroundingDINO (keep regions with >2% coverage)
2. Randomly generate a local editing plan (2-4 regions, random ops & intensities)
3. Execute edits via MaskedExecutor to produce ground-truth edited images
4. Save: config JSON + before/after stitched image + GT image
"""
import os
import json
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

from .utils import load_config


# Candidate regions and their default GroundingDINO prompts
REGION_PROMPTS = {
    "skin": "human skin area",
    "sky": "sky",
    "hair": "human hair",
    "background": "background area",
    "vegetation": "trees, grass, plants, foliage",
    "clothing": "clothing, clothes, fabric",
}

# Operations available for local edits
LOCAL_OPS = [
    "brightness", "contrast", "saturation", "temperature",
    "highlights", "shadows", "vibrance",
]


def detect_available_regions(image_path, mask_generator, min_coverage=0.02):
    """
    Detect which semantic regions are present in the image.
    Returns list of (region_name, mask_prompt) tuples with coverage > min_coverage.
    """
    from image_ops.non_gimp_ops import read_image

    image, _ = read_image(image_path)

    available = []
    for region, prompt in REGION_PROMPTS.items():
        try:
            mask = mask_generator.generate_mask(
                image, mask_type="semantic", mask_prompt=prompt
            )
            coverage = mask.mean()
            if coverage > min_coverage:
                available.append((region, prompt, coverage))
        except Exception as e:
            print(f"  Warning: failed to detect '{region}': {e}")

    return available


def generate_random_edit_plan(available_regions, num_edits=None):
    """
    Generate a random local editing plan.

    Args:
        available_regions: List of (region_name, mask_prompt, coverage) tuples.
        num_edits: Number of edits (default: random 2-4).

    Returns:
        List of dicts (JSON-serializable local edit specs).
    """
    if num_edits is None:
        num_edits = random.randint(2, min(4, len(available_regions)))

    selected = random.sample(available_regions, min(num_edits, len(available_regions)))

    plan = []
    for region_name, mask_prompt, _ in selected:
        op = random.choice(LOCAL_OPS)

        # Normal distribution for intensity, sigma=0.35, clipped to [-0.8, 0.8]
        value = float(np.clip(np.random.normal(0, 0.35), -0.8, 0.8))
        value = round(value, 2)

        # Skip near-zero values
        if abs(value) < 0.05:
            value = 0.15 * (1 if random.random() > 0.5 else -1)

        spec = {
            "region": region_name,
            "mask_type": "semantic",
            "mask_prompt": mask_prompt,
            "op": op,
            "value": value,
            "refine": {"feather": random.choice([3, 5, 8])},
            "reason_issue": "",
            "reason_solution": "",
        }
        plan.append(spec)

    return plan


def generate_puzzle_c_sample(
    image_path, mask_generator, executor, output_dir, sample_id
):
    """
    Generate a single Puzzle C sample:
    - Detect regions
    - Create random edit plan
    - Execute edits to produce GT
    - Save config + images
    """
    from image_ops.non_gimp_ops import read_image
    from local.local_config import parse_local_config

    # Detect available regions
    available = detect_available_regions(image_path, mask_generator)
    if len(available) < 2:
        print(f"  Skipping {image_path}: only {len(available)} regions detected.")
        return None

    # Generate plan
    plan = generate_random_edit_plan(available)

    # Prepare output paths
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"{sample_id}.json")
    gt_path = os.path.join(output_dir, f"{sample_id}_gt.tif")
    stitch_path = os.path.join(output_dir, f"{sample_id}_stitch.png")

    # Save config
    with open(config_path, "w") as f:
        json.dump(plan, f, indent=2)

    # Execute local edits to produce GT
    specs = parse_local_config(plan)
    executor.execute_local_edits(image_path, gt_path, specs)

    # Create before/after stitch for VLM querying
    _create_stitch(image_path, gt_path, stitch_path)

    return {
        "sample_id": sample_id,
        "source_image": image_path,
        "config_path": config_path,
        "gt_path": gt_path,
        "stitch_path": stitch_path,
        "plan": plan,
    }


def _create_stitch(before_path, after_path, output_path, max_size=800):
    """Create a side-by-side stitched image (before | after)."""
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")

    # Resize to common size
    w, h = before.size
    scale = min(max_size / w, max_size / h)
    if scale < 1:
        new_size = (int(w * scale), int(h * scale))
        before = before.resize(new_size, Image.LANCZOS)
        after = after.resize(new_size, Image.LANCZOS)

    w, h = before.size
    stitched = Image.new("RGB", (w * 2, h))
    stitched.paste(before, (0, 0))
    stitched.paste(after, (w, 0))
    stitched.save(output_path)


def generate_puzzle_c_dataset(
    range_a=0,
    range_b=None,
    config_path="configs/dataset_config.yaml",
):
    """
    Generate the full Puzzle C dataset from source images.
    """
    from local.mask_generator import MaskGenerator
    from local.masked_executor import MaskedExecutor

    config = load_config(config_path)

    source_pattern = config["image_sources"]["ppr10k_source"]
    output_dir = config.get("generation", {}).get("output_dirs", {}).get(
        "puzzle_c", "data/puzzle_c/configs"
    )
    images_output_dir = output_dir.replace("configs", "images")

    source_images = sorted(glob.glob(source_pattern))
    if range_b is None:
        range_b = len(source_images)
    source_images = source_images[range_a:range_b]

    print(f"Generating Puzzle C for {len(source_images)} images...")

    mask_gen = MaskGenerator()
    executor = MaskedExecutor()

    results = []
    for idx, img_path in enumerate(tqdm(source_images, desc="Puzzle C")):
        try:
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            sample_id = f"{image_name}_local"

            result = generate_puzzle_c_sample(
                img_path, mask_gen, executor, images_output_dir, sample_id
            )
            if result:
                # Save config separately
                config_save_path = os.path.join(output_dir, f"{sample_id}.json")
                os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
                with open(config_save_path, "w") as f:
                    json.dump(result["plan"], f, indent=2)
                results.append(result)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Cleanup
    executor.cleanup()

    print(f"Generated {len(results)} Puzzle C samples.")
    return results
