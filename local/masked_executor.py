"""
MaskedExecutor: applies local edits with mask-based blending.

For each LocalEditSpec:
  1. Generate mask (semantic or luminance_range) + apply refinements
  2. Apply the editing operation to a copy of the current image
  3. Blend edited copy with current image in float precision
"""
import os
import json
import tempfile
import numpy as np
from PIL import Image

from .mask_generator import MaskGenerator
from .mask_ops import apply_refinements
from .local_config import (
    LocalEditSpec,
    LOCAL_OP_TO_PIPELINE_OP,
    GIMP_REQUIRED_OPS,
)


class MaskedExecutor:
    """Executes local edits with mask-based region isolation and blending."""

    def __init__(self, config_path: str = "configs/local_editing_config.yaml"):
        self.mask_gen = MaskGenerator(config_path)
        defaults = self.mask_gen.config.get("mask_defaults", {})
        self._min_mask_coverage = float(defaults.get("min_mask_coverage", 0.0))
        self._temp_dir = tempfile.mkdtemp(prefix="monetgpt_local_")

    def execute_local_edits(
        self,
        src_image_path: str,
        output_path: str,
        specs: list,
        pipeline_config: dict = None,
    ) -> None:
        """
        Apply a sequence of local edits to the source image.

        Args:
            src_image_path: Path to the source image.
            output_path: Path to save the final result.
            specs: List of LocalEditSpec objects.
            pipeline_config: Pipeline config dict (for GIMP operations).
        """
        from image_ops.non_gimp_ops import read_image, save_tif

        image, norm_factor = read_image(src_image_path)
        current = image.copy()

        for i, spec in enumerate(specs):
            print(f"[Local Edit {i+1}/{len(specs)}] region={spec.region}, op={spec.op}, value={spec.value}")

            # 1. Generate mask
            mask = self._build_mask(current, spec)
            if mask.max() < 1e-6:
                print(f"  Warning: empty mask for region '{spec.region}', skipping.")
                continue

            coverage = float((mask > 0.01).mean())
            if coverage < self._min_mask_coverage:
                print(
                    f"  Warning: mask coverage {coverage:.4f} below minimum "
                    f"{self._min_mask_coverage:.4f} for region '{spec.region}', skipping."
                )
                continue

            # 2. Apply operation to a copy
            pipeline_op = LOCAL_OP_TO_PIPELINE_OP.get(spec.op.lower(), spec.op)
            edited = self._apply_op(
                current.copy(), pipeline_op, spec.value, norm_factor, pipeline_config
            )

            # 3. Blend in float space to avoid 8-bit quantization artifacts.
            current = self._blend_with_mask(current, edited, mask)

        # Save final result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith(".tif"):
            save_tif(current, norm_factor, output_path)
        else:
            out = np.clip(current, 0, 1)
            out = (out * 255).astype(np.uint8)
            Image.fromarray(out).save(output_path)

        print(f"[Local Edit] Saved result to {output_path}")

    def cleanup(self):
        """Remove temp files and unload GPU models."""
        import shutil

        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self.mask_gen.unload_models()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mask(self, image: np.ndarray, spec: LocalEditSpec) -> np.ndarray:
        """Generate and refine mask for a given spec."""
        # Build luminance_range params from refine dict if present
        lum_params = {}
        if spec.mask_type == "luminance_range":
            lum_params = {
                "lower": spec.refine.get("luminance_lower", 0.0),
                "upper": spec.refine.get("luminance_upper", 1.0),
                "softness": spec.refine.get("softness", 0.1),
            }

        mask = self.mask_gen.generate_mask(
            image,
            mask_type=spec.mask_type,
            mask_prompt=spec.mask_prompt,
            params=lum_params,
        )

        # Intersect semantic mask with luminance range if both specified
        if spec.mask_type == "semantic" and "luminance_range" in spec.refine:
            lum_range = spec.refine["luminance_range"]
            if isinstance(lum_range, list) and len(lum_range) == 2:
                lum_mask = self.mask_gen.generate_mask(
                    image,
                    mask_type="luminance_range",
                    params={"lower": lum_range[0], "upper": lum_range[1]},
                )
                mask = np.minimum(mask, lum_mask)

        refine = dict(spec.refine or {})
        defaults = self.mask_gen.config.get("mask_defaults", {})

        # Fill missing refinement defaults from config.
        if "feather" in refine and "feather_mode" not in refine:
            refine["feather_mode"] = defaults.get("feather_mode", "edge_inward")

        if spec.mask_type == "semantic":
            if "feather" not in refine and defaults.get("feather_sigma", 0) > 0:
                refine["feather"] = float(defaults.get("feather_sigma", 0))
                refine.setdefault(
                    "feather_mode", defaults.get("feather_mode", "edge_inward")
                )

            if "matting_refine" not in refine:
                refine["matting_refine"] = bool(defaults.get("matting_refine", True))
            if refine.get("matting_refine", False):
                refine.setdefault("trimap_radius", int(defaults.get("trimap_radius", 6)))
                refine.setdefault(
                    "trimap_threshold", float(defaults.get("trimap_threshold", 0.5))
                )
                refine.setdefault(
                    "matting_backend", str(defaults.get("matting_backend", "auto"))
                )
                refine.setdefault(
                    "matting_method", str(defaults.get("matting_method", "lkm"))
                )
                refine.setdefault(
                    "matting_model_id",
                    str(
                        defaults.get(
                            "matting_model_id",
                            "hustvl/vitmatte-small-composition-1k",
                        )
                    ),
                )
                refine.setdefault(
                    "matting_device", str(defaults.get("matting_device", "cuda"))
                )
                refine.setdefault(
                    "matting_fp16", bool(defaults.get("matting_fp16", True))
                )
                refine.setdefault(
                    "matting_use_roi", bool(defaults.get("matting_use_roi", True))
                )
                refine.setdefault(
                    "matting_roi_margin", int(defaults.get("matting_roi_margin", 32))
                )
                refine.setdefault(
                    "matting_max_dim", int(defaults.get("matting_max_dim", 1600))
                )
                refine.setdefault(
                    "matting_post_blur",
                    float(defaults.get("matting_post_blur", 0.0)),
                )
                refine.setdefault(
                    "feather_after_matting",
                    bool(defaults.get("feather_after_matting", False)),
                )

            contract_default = int(defaults.get("semantic_contract", 0))
            if contract_default > 0 and "contract" not in refine:
                refine["contract"] = contract_default

            if "guided_refine" not in refine:
                refine["guided_refine"] = bool(defaults.get("guided_refine", True))
            if refine.get("guided_refine", False):
                refine.setdefault("guided_radius", int(defaults.get("guided_radius", 8)))
                refine.setdefault("guided_eps", float(defaults.get("guided_eps", 1e-3)))

        # Apply refinements (feather, invert, etc.)
        mask = apply_refinements(mask, refine, guide_image=image)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def _apply_op(
        self,
        image: np.ndarray,
        pipeline_op: str,
        value: float,
        norm_factor: float,
        pipeline_config: dict = None,
    ) -> np.ndarray:
        """Apply a single editing operation to the full image."""
        if pipeline_op in GIMP_REQUIRED_OPS:
            return self._apply_gimp_op(image, pipeline_op, value, norm_factor, pipeline_config)
        else:
            return self._apply_non_gimp_op(image, pipeline_op, value, norm_factor)

    def _apply_non_gimp_op(
        self, image: np.ndarray, op: str, value: float, norm_factor: float
    ) -> np.ndarray:
        """Apply a non-GIMP operation directly in numpy."""
        from image_ops.non_gimp_ops import (
            adjust_exposure,
            adjust_tones,
            adjust_tint,
            adjust_vibrance,
            adjust_dehaze,
        )

        result = image.astype(np.float32, copy=True)

        if op == "Exposure":
            result = adjust_exposure(result, value)
        elif op == "Highlights":
            result = adjust_tones(result, highlights=value)
        elif op == "Shadows":
            result = adjust_tones(result, shadows=value)
        elif op == "Whites":
            result = adjust_tones(result, whites=value)
        elif op == "Tint":
            result = adjust_tint(result, value)
        elif op == "Vibrance":
            result = adjust_vibrance(result, value)
        elif op == "Dehaze":
            result = adjust_dehaze(result, norm_factor, value)
        else:
            # Fallback: try GIMP path
            return self._apply_gimp_op(image, op, value, norm_factor, None)

        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def _apply_gimp_op(
        self,
        image: np.ndarray,
        op: str,
        value: float,
        norm_factor: float,
        pipeline_config: dict = None,
    ) -> np.ndarray:
        """
        Apply a GIMP-only operation via temp file roundtrip:
        1. Save current image to temp .tif
        2. Write single-op config JSON
        3. Run GIMP subprocess
        4. Load result back
        """
        from image_ops.non_gimp_ops import save_tif, read_image
        from pipeline.utils import (
            update_pipeline_file_paths,
            execute_gimp_pipeline,
            load_combined_config,
        )

        # Prepare temp paths
        temp_input = os.path.join(self._temp_dir, "gimp_input.tif")
        temp_output = os.path.join(self._temp_dir, "gimp_output.tif")
        temp_config = os.path.join(self._temp_dir, "gimp_config.json")

        # Save image
        save_tif(image, norm_factor, temp_input)

        # Write single-op config (value in -100..100 scale)
        config_data = {op: int(value * 100)}
        with open(temp_config, "w") as f:
            json.dump(config_data, f)

        # Load pipeline config if not provided
        if pipeline_config is None:
            pipeline_config = load_combined_config()

        pipeline_file = pipeline_config.get("gimp", {}).get(
            "pipeline_file", "./image_ops/gimp_pipeline.py"
        )

        # Update GIMP pipeline paths and execute
        update_pipeline_file_paths(pipeline_file, temp_config, temp_input, temp_output)
        execute_gimp_pipeline(pipeline_config)

        # Load result back
        if os.path.exists(temp_output):
            result, _ = read_image(temp_output)
            return result
        else:
            print(f"  Warning: GIMP output not found at {temp_output}, returning original.")
            return image

    def _blend_with_mask(
        self,
        background: np.ndarray,
        overlay: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Blend overlay onto background using a float alpha mask."""
        alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
        bg = background.astype(np.float32, copy=False)
        ov = overlay.astype(np.float32, copy=False)
        result = bg * (1.0 - alpha) + ov * alpha
        return np.clip(result, 0.0, 1.0).astype(np.float32)
