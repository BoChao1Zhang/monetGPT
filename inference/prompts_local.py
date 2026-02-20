"""
Prompt templates for the local-editing stage (Stage 4).
Two-round VLM interaction:
  Round 1 (analysis): Identify region-level issues in the image.
  Round 2 (JSON): Output structured JSON array of local edit specs.
"""


def create_local_analysis_prompt(style_instruction: str = "", extra_instruction: str = "") -> str:
    """
    Create the analysis prompt for local/regional editing (Round 1).
    The VLM should identify region-specific issues (skin, sky, background, etc.).
    """
    prompt = f"""
Analyze the provided image and develop a professional-grade **local/regional editing plan**. Focus on region-specific issues that cannot be addressed by global adjustments alone.

Examine each semantic region of the image (such as skin, sky, hair, background, vegetation, clothing, shadows, highlights) and identify localized issues that need targeted correction.

For each regional issue found, provide:

**Region:** [Name the region, e.g., "skin", "sky", "background", "hair", "vegetation"]
**Issue:** [Describe the specific localized problem — e.g., "The skin tones appear overly warm and slightly oversaturated, making the subject look unnatural" or "The sky is underexposed and lacks vibrancy compared to the foreground"]
**Solution:** [Describe the targeted adjustment — e.g., "Reduce temperature and slightly decrease saturation specifically on the skin area" or "Increase brightness and saturation selectively on the sky region"]

Guidelines:
- Focus on 2-5 of the most impactful regional adjustments.
- Only suggest local edits where a global adjustment would negatively affect other regions.
- Consider luminance-based masking for highlight/shadow region isolation.
- Each adjustment should target ONE region with ONE operation for precise control.

{style_instruction}

{extra_instruction}
""".strip()
    return prompt


def create_local_json_prompt(style_instruction: str = "", extra_instruction: str = "") -> str:
    """
    Create the JSON generation prompt for local/regional editing (Round 2).
    The VLM should output a strict JSON array.
    """
    prompt = f"""
Based on your regional editing analysis and the original image, output the **optimal** local adjustments as a JSON array. Each element represents one targeted regional edit.

**Output format** (strict JSON array, 2-5 elements):
```json
[
  {{
    "region": "skin",
    "mask_type": "semantic",
    "mask_prompt": "human skin area",
    "op": "saturation",
    "value": -0.15,
    "refine": {{"feather": 5}},
    "reason_issue": "Skin tones are overly saturated, appearing unnatural.",
    "reason_solution": "Slightly reduce saturation on skin to restore a natural look."
  }},
  {{
    "region": "sky",
    "mask_type": "semantic",
    "mask_prompt": "sky",
    "op": "brightness",
    "value": 0.20,
    "refine": {{"feather": 8}},
    "reason_issue": "Sky region is underexposed compared to the foreground.",
    "reason_solution": "Increase brightness on the sky to balance exposure across the frame."
  }}
]
```

**Field specifications:**
- `region`: Semantic name (skin, sky, hair, background, vegetation, clothing, shadows, highlights, midtones)
- `mask_type`: Either `"semantic"` (for object/region detection) or `"luminance_range"` (for tonal range isolation)
- `mask_prompt`: Text description for semantic detection (e.g., "human skin area", "sky", "trees and grass"). Leave empty for luminance_range.
- `op`: One of: brightness, contrast, saturation, temperature, tint, vibrance, highlights, shadows, whites, blacks, dehaze, exposure
- `value`: Float in [-1.0, +1.0]. Positive = increase, negative = decrease. Use moderate values (-0.5 to +0.5) unless the issue is severe.
- `refine`: Optional refinements:
  - `"feather"`: Gaussian blur sigma for smooth mask edges (recommended: 3-10)
  - `"invert"`: true to invert the mask (edit everything EXCEPT the detected region)
  - `"luminance_range"`: [lower, upper] to intersect semantic mask with a luminance range (e.g., [0.6, 1.0] for highlights only within the region)
  - `"luminance_lower"`, `"luminance_upper"`, `"softness"`: For luminance_range mask_type
- `reason_issue`: Brief description of the localized issue.
- `reason_solution`: Brief description of how this adjustment resolves it.

**Rules:**
- Output ONLY the JSON array, no surrounding text or markdown code fences.
- Limit to 2-5 adjustments — focus on the most impactful ones.
- Each adjustment should target one region with one operation.
- Ensure values are proportional to the severity of the issue.

{style_instruction}

{extra_instruction}
""".strip()
    return prompt
