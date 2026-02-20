"""
Core editing pipeline that combines GIMP and non-GIMP operations.
"""
import os
import json
from typing import Optional
from .utils import (
    load_combined_config,
    update_pipeline_file_paths,
    execute_gimp_pipeline,
    ensure_directory
)
from image_ops.non_gimp_ops import execute_non_gimp_pipeline as _execute_non_gimp


def execute_non_gimp_pipeline(config_path: str, src_path: str, output_path: str):
    """Import and execute non-GIMP pipeline."""
    # Import here to avoid circular imports
    import sys
    sys.path.append('.')

    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))

    return _execute_non_gimp(config_path, src_path, output_path)


class ImageEditingPipeline:
    """Main pipeline for applying image editing operations."""

    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config = load_combined_config(config_path)
        self.pipeline_file = self.config["gimp"]["pipeline_file"]
        self._local_executor = None

    def execute_edit(
        self,
        config_path: str,
        src_image_path: str,
        output_path: str
    ) -> None:
        """Execute complete editing pipeline (non-GIMP + GIMP operations, or local edits)."""
        # Load config to check if it's a local-editing config (JSON array)
        with open(config_path) as f:
            config_data = json.load(f)

        if isinstance(config_data, list):
            # Local-editing stage: dispatch to MaskedExecutor
            self._execute_local_edit(config_data, src_image_path, output_path)
        else:
            # Original global pipeline (unchanged)
            execute_non_gimp_pipeline(config_path, src_image_path, output_path)
            if self._requires_gimp(config_data):
                update_pipeline_file_paths(
                    self.pipeline_file,
                    config_path,
                    output_path,
                    output_path
                )
                ok = execute_gimp_pipeline(self.config)
                if not ok:
                    print(
                        "Warning: GIMP stage failed/timed out. "
                        "Keeping non-GIMP result for this step."
                    )

    def _requires_gimp(self, config_data: dict) -> bool:
        """
        Return True if this stage contains non-zero operations that require GIMP.
        """
        from local.local_config import GIMP_REQUIRED_OPS

        for op_name, value in config_data.items():
            if op_name not in GIMP_REQUIRED_OPS:
                continue
            try:
                if float(value) != 0.0:
                    return True
            except (TypeError, ValueError):
                # Conservative fallback for non-numeric values.
                return True
        return False

    def _execute_local_edit(
        self,
        config_data: list,
        src_image_path: str,
        output_path: str,
    ) -> None:
        """Execute local/regional editing via MaskedExecutor."""
        from local.local_config import parse_local_config

        specs = parse_local_config(config_data)
        executor = self._get_local_executor()
        executor.execute_local_edits(
            src_image_path, output_path, specs,
            pipeline_config=self.config,
        )

    def _get_local_executor(self):
        """
        Reuse a single MaskedExecutor instance per pipeline process.
        This avoids reloading GroundingDINO/SAM2 for every local-edit call.
        """
        if self._local_executor is None:
            from local.masked_executor import MaskedExecutor

            self._local_executor = MaskedExecutor()
        return self._local_executor

    def warmup_local_models(self):
        """
        Proactively load semantic-mask models.
        Useful for moving cold-start latency to session initialization.
        """
        executor = self._get_local_executor()
        # Lazy model loaders are private to MaskGenerator, but warming here keeps
        # per-operation latency low in interactive sessions.
        executor.mask_gen._load_grounding_dino()
        executor.mask_gen._load_sam2()

        defaults = executor.mask_gen.config.get("mask_defaults", {})
        if bool(defaults.get("matting_refine", False)):
            from local.mask_ops import warmup_matting_backend

            warmup_matting_backend(
                backend=str(defaults.get("matting_backend", "auto")),
                model_id=str(
                    defaults.get(
                        "matting_model_id", "hustvl/vitmatte-small-composition-1k"
                    )
                ),
                device=str(defaults.get("matting_device", "cuda")),
                use_fp16=bool(defaults.get("matting_fp16", True)),
            )

    def cleanup(self):
        """Release local executor resources (models/temp files)."""
        if self._local_executor is not None:
            self._local_executor.cleanup()
            self._local_executor = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
    
    def execute_single_stage(
        self,
        adjustments,
        src_path: str,
        output_path: str,
        is_local: bool = False,
    ) -> None:
        """
        Execute a single editing stage — convenience wrapper for agent executor.

        Args:
            adjustments: For global edits: dict of {op_name: value}.
                        For local edits: list of local edit spec dicts.
            src_path: Source image path.
            output_path: Output image path.
            is_local: If True, treat adjustments as local edit specs (JSON array).
        """
        import tempfile

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Write temp config file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=os.path.dirname(output_path) or "."
        ) as f:
            json.dump(adjustments if is_local else adjustments, f, indent=2)
            temp_config = f.name

        try:
            self.execute_edit(temp_config, src_path, output_path)
        finally:
            if os.path.exists(temp_config):
                os.remove(temp_config)

    def execute_gimp_only(
        self,
        config_path: str,
        src_image_path: str, 
        output_path: str
    ) -> None:
        """Execute only GIMP operations."""
        update_pipeline_file_paths(
            self.pipeline_file,
            config_path,
            src_image_path,
            output_path
        )
        execute_gimp_pipeline(self.config)
    
    def execute_non_gimp_only(
        self,
        config_path: str,
        src_image_path: str,
        output_path: str
    ) -> None:
        """Execute only non-GIMP operations."""
        execute_non_gimp_pipeline(config_path, src_image_path, output_path)
