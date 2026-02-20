"""
Mask generation: semantic AI masks (GroundingDINO + SAM2) and luminance-range masks.
Models are lazily loaded on first use and can be explicitly unloaded.
"""
import numpy as np
import yaml


class MaskGenerator:
    """Generates semantic and luminance-range masks for local editing."""

    def __init__(self, config_path: str = "configs/local_editing_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self._sam2_predictor = None
        self._grounding_dino_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_mask(
        self,
        image: np.ndarray,
        mask_type: str,
        mask_prompt: str = "",
        params: dict = None,
    ) -> np.ndarray:
        """
        Generate a mask of the requested type.

        Args:
            image: (H, W, 3) float32 [0, 1] RGB image.
            mask_type: "semantic" or "luminance_range".
            mask_prompt: Text prompt for semantic masks (e.g. "human skin area").
            params: Extra parameters (e.g. {"lower": 0.6, "upper": 1.0, "softness": 0.1}).

        Returns:
            (H, W) float32 [0, 1] mask.
        """
        if params is None:
            params = {}

        if mask_type == "semantic":
            return self._generate_semantic_mask(image, mask_prompt)
        elif mask_type == "luminance_range":
            return self._generate_luminance_range_mask(image, params)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

    def unload_models(self):
        """Release GPU memory for SAM2 and GroundingDINO."""
        import torch

        if self._sam2_predictor is not None:
            del self._sam2_predictor
            self._sam2_predictor = None
        if self._grounding_dino_model is not None:
            del self._grounding_dino_model
            self._grounding_dino_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Semantic mask (GroundingDINO → SAM2)
    # ------------------------------------------------------------------

    def _load_grounding_dino(self):
        """Lazily load GroundingDINO model."""
        if self._grounding_dino_model is not None:
            return

        import torch
        from groundingdino.util.inference import load_model

        cfg = self.config["grounding_dino"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._grounding_dino_model = load_model(
            cfg["config_path"], cfg["checkpoint_path"], device=device
        )

    def _load_sam2(self):
        """Lazily load SAM2 predictor."""
        if self._sam2_predictor is not None:
            return

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        cfg = self.config["sam2"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(
            cfg["model_config"], cfg["checkpoint_path"], device=device
        )
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)

    def _generate_semantic_mask(
        self, image: np.ndarray, mask_prompt: str
    ) -> np.ndarray:
        """
        Generate semantic mask using GroundingDINO (text→boxes) + SAM2 (boxes→masks).
        """
        import torch
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import predict

        if not mask_prompt or not mask_prompt.strip():
            H, W = image.shape[:2]
            print("  Warning: empty semantic mask prompt, returning empty mask.")
            return np.zeros((H, W), dtype=np.float32)

        self._load_grounding_dino()
        self._load_sam2()

        cfg = self.config["grounding_dino"]
        box_threshold = cfg.get("box_threshold", 0.3)
        text_threshold = cfg.get("text_threshold", 0.25)

        # GroundingDINO expects a transformed torch tensor.
        from PIL import Image as PILImage

        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(image_uint8)
        dino_transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = dino_transform(pil_image, None)

        # Predict bounding boxes
        boxes, logits, phrases = predict(
            model=self._grounding_dino_model,
            image=image_tensor,
            caption=mask_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        H, W = image.shape[:2]

        if len(boxes) == 0:
            # No detection — return empty mask
            return np.zeros((H, W), dtype=np.float32)

        # GroundingDINO predict() returns normalized boxes in cxcywh format.
        # Convert to absolute xyxy and clamp to image bounds for SAM2.
        boxes_abs = boxes.clone().float()
        cx = boxes_abs[:, 0] * W
        cy = boxes_abs[:, 1] * H
        bw = boxes_abs[:, 2] * W
        bh = boxes_abs[:, 3] * H

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        boxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes_abs[:, 0] = torch.clamp(boxes_abs[:, 0], 0, W - 1)
        boxes_abs[:, 1] = torch.clamp(boxes_abs[:, 1], 0, H - 1)
        boxes_abs[:, 2] = torch.clamp(boxes_abs[:, 2], 0, W - 1)
        boxes_abs[:, 3] = torch.clamp(boxes_abs[:, 3], 0, H - 1)

        # SAM2 prediction
        self._sam2_predictor.set_image(image_uint8)

        input_boxes = boxes_abs.cpu().numpy()
        masks_list = []

        for box in input_boxes:
            masks, scores, _ = self._sam2_predictor.predict(
                box=box,
                multimask_output=True,
            )
            # Take the mask with the highest score
            best_idx = np.argmax(scores)
            masks_list.append(masks[best_idx])

        # Union all masks
        combined = np.zeros((H, W), dtype=np.float32)
        for m in masks_list:
            combined = np.maximum(combined, m.astype(np.float32))

        return self._postprocess_semantic_mask(combined)

    # ------------------------------------------------------------------
    # Luminance range mask (reuses non_gimp_ops logic)
    # ------------------------------------------------------------------

    def _generate_luminance_range_mask(
        self, image: np.ndarray, params: dict
    ) -> np.ndarray:
        """
        Generate luminance-range mask reusing the smoothstep/create_mask logic
        from image_ops/non_gimp_ops.py.
        """
        from image_ops.non_gimp_ops import create_mask

        # Compute luminance
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        L = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

        lower = float(params.get("lower", 0.0))
        upper = float(params.get("upper", 1.0))
        softness = float(params.get("softness", 0.1))

        return create_mask(L, lower, upper, softness)

    def _postprocess_semantic_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean semantic masks to reduce noisy islands / interior holes that create
        uneven local edits.
        """
        from scipy.ndimage import binary_fill_holes, label

        defaults = self.config.get("mask_defaults", {})
        threshold = float(defaults.get("semantic_threshold", 0.5))
        min_component_ratio = float(defaults.get("min_component_ratio", 0.0005))

        binary = mask >= threshold
        if not np.any(binary):
            return np.zeros_like(mask, dtype=np.float32)

        # Fill interior holes to avoid patchy opacity in the selected region.
        binary = binary_fill_holes(binary)

        if min_component_ratio > 0:
            labels, num_labels = label(binary)
            if num_labels > 0:
                min_pixels = max(1, int(binary.size * min_component_ratio))
                keep = np.zeros_like(binary, dtype=bool)
                for idx in range(1, num_labels + 1):
                    component = labels == idx
                    if int(component.sum()) >= min_pixels:
                        keep |= component
                if np.any(keep):
                    binary = keep

        return binary.astype(np.float32)
