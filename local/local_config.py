"""
Local editing configuration: dataclasses, parsing, and operation mapping.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class LocalEditSpec:
    """Specification for a single local/regional edit."""
    region: str                    # "skin", "sky", "background", etc.
    mask_type: str                 # "semantic" or "luminance_range"
    mask_prompt: str               # GroundingDINO text prompt (for semantic masks)
    op: str                        # local op name: "brightness", "contrast", etc.
    value: float                   # [-1.0, 1.0]
    refine: dict = field(default_factory=dict)  # {"feather": 5, "invert": true, ...}
    reason_issue: str = ""
    reason_solution: str = ""


# Map local op names → pipeline operation names used in non_gimp_ops / gimp_ops
LOCAL_OP_TO_PIPELINE_OP = {
    "brightness": "Exposure",
    "exposure": "Exposure",
    "contrast": "Contrast",
    "saturation": "Saturation",
    "temperature": "Temperature",
    "tint": "Tint",
    "vibrance": "Vibrance",
    "highlights": "Highlights",
    "shadows": "Shadows",
    "whites": "Whites",
    "blacks": "Blacks",
    "dehaze": "Dehaze",
}

# Operations that must be executed through GIMP subprocess
GIMP_REQUIRED_OPS = {"Saturation", "Contrast", "Temperature", "Blacks"}

# Operations handled directly by non_gimp_ops (numpy-based)
NON_GIMP_OPS = {"Exposure", "Highlights", "Shadows", "Whites", "Tint", "Vibrance", "Dehaze"}
VALID_MASK_TYPES = {"semantic", "luminance_range"}
VALID_LOCAL_OPS = set(LOCAL_OP_TO_PIPELINE_OP.keys())


def is_local_config(config_data) -> bool:
    """Check if config_data represents a local-editing configuration (JSON array)."""
    return isinstance(config_data, list)


def parse_local_config(config_list: list) -> List[LocalEditSpec]:
    """Parse a JSON array of local edit specs into LocalEditSpec objects."""
    specs = []
    if not isinstance(config_list, list):
        print("Warning: local config is not a list, skipping.")
        return specs

    for idx, item in enumerate(config_list):
        if not isinstance(item, dict):
            print(f"Warning: local spec #{idx} is not a JSON object, skipping.")
            continue

        mask_type = str(item.get("mask_type", "semantic")).strip().lower()
        if mask_type not in VALID_MASK_TYPES:
            print(
                f"Warning: local spec #{idx} has invalid mask_type '{mask_type}', skipping."
            )
            continue

        op = str(item.get("op", "")).strip().lower()
        if op not in VALID_LOCAL_OPS:
            print(f"Warning: local spec #{idx} has invalid op '{op}', skipping.")
            continue

        try:
            value = float(item.get("value", 0))
        except (TypeError, ValueError):
            print(f"Warning: local spec #{idx} has non-numeric value, skipping.")
            continue

        if value < -1.0 or value > 1.0:
            print(
                f"Warning: local spec #{idx} value {value} out of range [-1, 1], clamping."
            )
            value = max(-1.0, min(1.0, value))

        refine = item.get("refine", {})
        if not isinstance(refine, dict):
            print(f"Warning: local spec #{idx} refine is not an object, ignoring refine.")
            refine = {}

        region = str(item.get("region", "unknown")).strip() or "unknown"
        mask_prompt = str(item.get("mask_prompt", "")).strip()
        if mask_type == "semantic" and not mask_prompt:
            mask_prompt = region
            print(
                f"Warning: local spec #{idx} missing semantic mask_prompt, "
                f"falling back to region '{region}'."
            )

        spec = LocalEditSpec(
            region=region,
            mask_type=mask_type,
            mask_prompt=mask_prompt,
            op=op,
            value=value,
            refine=refine,
            reason_issue=str(item.get("reason_issue", "")),
            reason_solution=str(item.get("reason_solution", "")),
        )
        specs.append(spec)
    return specs
