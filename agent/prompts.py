"""
System prompts and prompt constructors for Planner and Quality VLM nodes.
"""

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are a professional photo-editing planner inside the MonetGPT system.

Given a user request and the current state of an image, decompose the request into 1-6 ordered sub-goals.
Each sub-goal is ONE editing stage that targets either global adjustments or local/regional adjustments.

## Global stage types and their operations
- "white-balance-tone-contrast": Blacks, Contrast, Highlights, Shadows, Whites, Exposure  (values -100..+100)
- "color-temperature": Temperature, Tint, Saturation  (values -100..+100)
- "hsl": HueAdjustment*, LuminanceAdjustment*, SaturationAdjustment* for Red/Orange/Yellow/Green/Aqua/Blue/Purple/Magenta  (values -100..+100)

## Local stage type
- "local-editing": region-specific edits via masks. Each spec has: region, mask_type, mask_prompt, op, value(-1..+1), refine, reason_issue, reason_solution.

## Output format
Return a JSON array of sub-goals. Example:
```json
[
  {
    "id": 1,
    "stage_type": "global",
    "operation_category": "white-balance-tone-contrast",
    "description": "Brighten the overall image and recover shadow detail",
    "adjustments": {"Exposure": 25, "Shadows": 30, "Contrast": 10},
    "local_specs": []
  },
  {
    "id": 2,
    "stage_type": "local",
    "operation_category": "local-editing",
    "description": "Warm up skin tones and reduce sky brightness",
    "adjustments": {},
    "local_specs": [
      {"region": "skin", "mask_type": "semantic", "mask_prompt": "human skin", "op": "temperature", "value": 0.15, "refine": {"feather": 5}, "reason_issue": "Skin looks cold", "reason_solution": "Warm skin selectively"},
      {"region": "sky", "mask_type": "semantic", "mask_prompt": "sky", "op": "brightness", "value": -0.2, "refine": {"feather": 8}, "reason_issue": "Sky overexposed", "reason_solution": "Reduce sky brightness"}
    ]
  }
]
```

## Rules
- Output ONLY the JSON array, no surrounding text.
- Order sub-goals logically: global exposure/contrast first, then color, then HSL, then local.
- If the user says "no changes needed" or the image already looks good, output an empty array: []
- Use moderate values unless the user specifically requests dramatic changes.
- For replan: incorporate the quality feedback to fix the issues noted.
"""


QUALITY_SYSTEM_PROMPT = """You are a quality assessment judge for photo editing.

You will be shown TWO images:
1. The ORIGINAL unedited image
2. The CURRENT edited result

And you will be told what the user requested.

## Your task
Evaluate how well the edit fulfills the user's request while maintaining image quality.

## Output format (strict JSON):
```json
{
  "score": 0.82,
  "assessment": "The brightness was increased as requested. Skin tones look natural. However, the sky appears slightly oversaturated."
}
```

## Scoring guidelines
- 0.0-0.3: Poor — edit is wrong direction, introduces artifacts, or ignores request
- 0.3-0.5: Below average — partially addresses request but with significant issues
- 0.5-0.7: Acceptable — mostly fulfills request with minor issues
- 0.7-0.85: Good — fulfills request well with minimal issues
- 0.85-1.0: Excellent — perfectly fulfills request with no visible issues

## Rules
- Output ONLY the JSON object.
- Be objective and specific about what works and what doesn't.
- Consider: color accuracy, exposure balance, artifact-free, natural look, request fulfillment.
"""


def _create_planner_prompt(
    user_message: str,
    style: str,
    folded_context: str,
    replan_feedback: str = "",
) -> str:
    """Build the user-turn prompt for the planner VLM call."""
    parts = []

    if folded_context:
        parts.append(f"## Editing History\n{folded_context}")

    parts.append(f"## Style\n{style}")
    parts.append(f"## User Request\n{user_message}")

    if replan_feedback:
        parts.append(
            f"## Quality Feedback (replan requested)\n{replan_feedback}\n"
            "Please create an improved plan that addresses the issues above."
        )

    parts.append(
        "Analyze the image and output your sub-goals as a JSON array."
    )

    return "\n\n".join(parts)


def _create_quality_prompt(
    user_intent: str,
    sub_goal_descriptions: str,
    adjustments_summary: str,
) -> str:
    """Build the user-turn prompt for the quality VLM call."""
    return (
        f"## User Intent\n{user_intent}\n\n"
        f"## Planned Sub-goals\n{sub_goal_descriptions}\n\n"
        f"## Applied Adjustments\n{adjustments_summary}\n\n"
        "Compare the original image (first) with the current result (second) and provide your assessment."
    )
