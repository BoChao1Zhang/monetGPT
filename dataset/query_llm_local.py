"""
LLM query functions for generating Puzzle C (local editing) reasoning.

Same pattern as query_llm.py: send ground-truth local edits + stitched image
to Gemini to generate high-quality reasoning for each regional adjustment.
"""
import json
import glob
import os
import argparse
from tqdm import tqdm
from .utils import load_config, create_openai_client, encode_image, send_request_with_retry


def format_local_plan_text(plan):
    """Convert local edit plan to human-readable text for the LLM prompt."""
    parts = []
    for i, spec in enumerate(plan, 1):
        value_str = f"+{spec['value']}" if spec['value'] > 0 else f"{spec['value']}"
        parts.append(
            f"  {i}. Region: {spec['region']}, Operation: {spec['op']}, "
            f"Value: {value_str}, Mask: {spec['mask_type']}"
        )
    return "\n".join(parts)


def send_request_puzzle_c(image_path, plan, config):
    """Send request for Puzzle C reasoning generation."""
    client = create_openai_client(config)
    original_image = encode_image(image_path)

    plan_text = format_local_plan_text(plan)

    instruction = f"""
Examine the side-by-side comparison image (original on the left, edited on the right). The edited image was produced by applying **local/regional adjustments** — each targeting a specific semantic area of the image rather than applying global changes.

The actual local adjustments applied by the professional editor are:
{plan_text}

Values range from -1.0 to +1.0. Each adjustment targets a specific region using mask-based isolation.

Your task:
1. For EACH regional adjustment listed above, explain:
   - **Region & Issue:** What specific problem exists in that region of the ORIGINAL image? Be precise — reference visual elements, lighting, color, texture.
   - **Solution:** How does the targeted adjustment resolve this issue? Describe the expected improvement in future tense.

2. Explain WHY a local/regional approach is necessary here — why a global adjustment would not suffice (e.g., brightening the sky globally would overexpose the foreground).

3. Conclude with a brief summary of how all regional edits work together to achieve a cohesive, professional result.

**Rules:**
- Do NOT mention numerical values. Use descriptive intensity: Very Slight, Slight, Mild, Moderate, Noticeable, Significant, Very Significant, Extremely Intense.
- Be specific about visual elements in the image.
- Use future tense for solutions ("the edited image will have...").
- Reference the actual regions visible in the image.

Start your response with:
**Local/Regional Editing Analysis**
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an advanced image-editing assistant with expertise in Adobe Lightroom "
                "and local/regional photo editing. You analyze stitched before/after images and "
                "provide detailed reasoning for regional adjustments."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{original_image}",
                    },
                },
            ],
        },
    ]

    result = client.chat.completions.create(
        messages=messages,
        model=config["model"],
    )
    return result.choices[0].message.content


def query_puzzle_c(range_a, range_b, config_path="configs/dataset_config.yaml"):
    """Query LLM for Puzzle C reasoning."""
    config = load_config(config_path)
    puzzle_config = config["puzzles"]["puzzle_c"]

    stitch_pattern = puzzle_config["images_query_path"]
    reasoning_base_path = puzzle_config["reasoning_path"].replace("/*.txt", "")
    configs_base_path = config["generation"]["output_dirs"]["puzzle_c"]

    images = sorted(glob.glob(stitch_pattern))

    for image in tqdm(images[range_a:range_b], desc="Querying Puzzle C"):
        try:
            image_filename = os.path.splitext(os.path.basename(image))[0]
            # Remove _stitch suffix to get sample_id
            sample_id = image_filename.replace("_stitch", "")

            reasoning_save_path = f"{reasoning_base_path}/{sample_id}.txt"

            if os.path.exists(reasoning_save_path):
                continue

            os.makedirs(os.path.dirname(reasoning_save_path), exist_ok=True)

            # Load the local edit plan
            config_file = f"{configs_base_path}/{sample_id}.json"
            if not os.path.exists(config_file):
                print(f"Config not found: {config_file}")
                continue

            with open(config_file, "r") as f:
                plan = json.load(f)

            output = send_request_with_retry(
                send_request_puzzle_c,
                image, plan, config,
                config=config,
            )

            with open(reasoning_save_path, "w", encoding="utf-8") as f:
                f.write(output)

        except Exception as e:
            print(f"Error processing {image}: {e}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Query LLM for Puzzle C reasoning.")
    parser.add_argument("range_a", type=int, help="Start index.")
    parser.add_argument("range_b", type=int, help="End index.")
    parser.add_argument("--config", type=str, default="configs/dataset_config.yaml")
    args = parser.parse_args()

    query_puzzle_c(args.range_a, args.range_b, args.config)


if __name__ == "__main__":
    main()
