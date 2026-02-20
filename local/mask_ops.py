"""
Mask composition and refinement operations.
Uses numpy/scipy and optional OpenCV ximgproc guided filtering.
All masks are (H, W) float32 arrays in [0, 1].
"""

import threading
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


_VITMATTE_CACHE = {}
_VITMATTE_LOCK = threading.Lock()


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """Clamp and cast to float32 [0, 1]."""
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _as_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a soft mask to binary support."""
    return _normalize_mask(mask) >= float(threshold)


def refine_contract(mask: np.ndarray, pixels: int = 1) -> np.ndarray:
    """
    Contract a mask by eroding its binary support.
    Useful for reducing halo caused by mask spill over strong boundaries.
    """
    if pixels <= 0:
        return _normalize_mask(mask)

    import cv2

    binary = _as_binary(mask).astype(np.uint8)
    kernel_size = int(2 * pixels + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    contracted = cv2.erode(binary, kernel, iterations=1)
    return contracted.astype(np.float32)


def refine_feather_edge(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """
    Symmetric edge feathering:
    - Keeps interior close to 1 and exterior close to 0.
    - Smooths both sides of the boundary.
    """
    if sigma <= 0:
        return _normalize_mask(mask)

    binary = _as_binary(mask)
    if not np.any(binary):
        return np.zeros_like(mask, dtype=np.float32)

    width = max(float(sigma), 1.0)
    dist_in = distance_transform_edt(binary)
    dist_out = distance_transform_edt(~binary)
    signed_distance = dist_in - dist_out
    feathered = np.clip(0.5 + (signed_distance / (2.0 * width)), 0.0, 1.0)
    return _normalize_mask(feathered)


def refine_feather_edge_inward(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """
    Inward-only feathering:
    - Outside stays 0 to avoid color spill/halo.
    - Transition band lives only inside the selected region.
    """
    if sigma <= 0:
        return _normalize_mask(mask)

    binary = _as_binary(mask)
    if not np.any(binary):
        return np.zeros_like(mask, dtype=np.float32)

    width = max(float(sigma), 1.0)
    dist_in = distance_transform_edt(binary)
    feathered = np.clip(dist_in / width, 0.0, 1.0)
    feathered = np.where(binary, feathered, 0.0)
    return _normalize_mask(feathered)


def refine_feather_gaussian(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Classic Gaussian feathering (legacy behavior)."""
    if sigma <= 0:
        return _normalize_mask(mask)
    result = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return _normalize_mask(result)


def _build_trimap(binary: np.ndarray, radius: int) -> np.ndarray:
    """
    Build a trimap from a binary mask:
    - known foreground: 1.0
    - known background: 0.0
    - unknown boundary band: 0.5
    """
    import cv2

    r = max(1, int(radius))
    kernel_size = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    fg = cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    bg = ~cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)

    trimap = np.full(binary.shape, 0.5, dtype=np.float32)
    trimap[bg] = 0.0
    trimap[fg] = 1.0
    return trimap


def _resize_pair_for_matting(
    image: np.ndarray, trimap: np.ndarray, max_dim: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Resize image/trimap for faster matting and return whether resized."""
    if max_dim <= 0:
        return image, trimap, False

    import cv2

    h, w = trimap.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, trimap, False

    scale = float(max_dim) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    trimap_small = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image_small, trimap_small, True


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return xyxy bbox for a boolean mask, or None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def _crop_to_unknown_roi(
    image: np.ndarray, trimap: np.ndarray, margin: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Crop image + trimap to unknown-band ROI for faster matting.
    Returns cropped image, cropped trimap, and xyxy ROI in original coords.
    """
    h, w = trimap.shape[:2]
    unknown = trimap == 0.5
    bbox = _bbox_from_mask(unknown)
    if bbox is None:
        return image, trimap, (0, 0, w, h)

    x1, y1, x2, y2 = bbox
    m = max(1, int(margin))
    x1 = max(0, x1 - m)
    y1 = max(0, y1 - m)
    x2 = min(w, x2 + m)
    y2 = min(h, y2 + m)
    return image[y1:y2, x1:x2], trimap[y1:y2, x1:x2], (x1, y1, x2, y2)


def _get_vitmatte_bundle(
    model_id: str,
    device_preference: str = "cuda",
    use_fp16: bool = True,
):
    """
    Lazy-load and cache ViTMatte model + processor.
    Cache key includes model_id/device/fp16 to support multiple configs.
    """
    import torch
    from transformers import VitMatteImageProcessor, VitMatteForImageMatting

    pref = str(device_preference or "cuda").lower()
    if pref == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif pref == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fp16 = bool(use_fp16 and device == "cuda")
    key = (str(model_id), device, fp16)

    with _VITMATTE_LOCK:
        cached = _VITMATTE_CACHE.get(key)
        if cached is not None:
            return cached

        processor = VitMatteImageProcessor.from_pretrained(model_id)
        model = VitMatteForImageMatting.from_pretrained(model_id)
        model = model.to(device).eval()
        bundle = {
            "processor": processor,
            "model": model,
            "device": device,
            "fp16": fp16,
        }
        _VITMATTE_CACHE[key] = bundle
        return bundle


def warmup_matting_backend(
    backend: str = "auto",
    model_id: str = "hustvl/vitmatte-small-composition-1k",
    device: str = "cuda",
    use_fp16: bool = True,
) -> None:
    """Warm up selected matting backend (no-op for CPU backend)."""
    backend_name = str(backend or "auto").strip().lower()
    if backend_name == "cpu":
        return
    _get_vitmatte_bundle(
        model_id=model_id,
        device_preference=device,
        use_fp16=use_fp16,
    )


def _estimate_alpha_vitmatte(
    image: np.ndarray,
    trimap: np.ndarray,
    model_id: str,
    device: str,
    use_fp16: bool,
    max_dim: int,
) -> np.ndarray:
    """Estimate alpha using ViTMatte (GPU preferred)."""
    from PIL import Image as PILImage
    import torch

    image_m, trimap_m, resized = _resize_pair_for_matting(image, trimap, max_dim=max_dim)
    bundle = _get_vitmatte_bundle(
        model_id=model_id,
        device_preference=device,
        use_fp16=use_fp16,
    )
    processor = bundle["processor"]
    model = bundle["model"]
    model_device = bundle["device"]
    fp16 = bundle["fp16"]

    image_u8 = (np.clip(image_m, 0.0, 1.0) * 255.0).astype(np.uint8)
    trimap_u8 = (np.clip(trimap_m, 0.0, 1.0) * 255.0).astype(np.uint8)
    image_pil = PILImage.fromarray(image_u8, mode="RGB")
    trimap_pil = PILImage.fromarray(trimap_u8, mode="L")

    inputs = processor(images=image_pil, trimaps=trimap_pil, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.inference_mode():
        if fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                alphas = model(**inputs).alphas
        else:
            alphas = model(**inputs).alphas

    alpha = alphas[0, 0].detach().float().cpu().numpy()
    alpha = _normalize_mask(alpha)

    # Processor/model may return padded size; resize back to trimap working size.
    if alpha.shape[:2] != trimap_m.shape[:2]:
        import cv2

        th, tw = trimap_m.shape[:2]
        alpha = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )
        alpha = _normalize_mask(alpha)

    if resized:
        import cv2

        h, w = trimap.shape[:2]
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )

    return _normalize_mask(alpha)


def _estimate_alpha_pymatting(
    image: np.ndarray,
    trimap: np.ndarray,
    method: str,
    max_dim: int,
) -> np.ndarray:
    """Estimate alpha using CPU pymatting backends."""
    try:
        from pymatting import estimate_alpha_cf, estimate_alpha_knn, estimate_alpha_lkm
    except Exception:
        raise RuntimeError("pymatting is not installed")

    image_m, trimap_m, resized = _resize_pair_for_matting(image, trimap, max_dim=max_dim)

    method_name = str(method or "lkm").strip().lower()
    if method_name == "cf":
        alpha = estimate_alpha_cf(image_m, trimap_m)
    elif method_name == "knn":
        alpha = estimate_alpha_knn(image_m, trimap_m)
    else:
        alpha = estimate_alpha_lkm(image_m, trimap_m)

    alpha = _normalize_mask(alpha)
    if resized:
        import cv2

        h, w = trimap.shape[:2]
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )
    return _normalize_mask(alpha)


def refine_trimap_matting(
    mask: np.ndarray,
    guide_image: np.ndarray,
    trimap_radius: int = 6,
    threshold: float = 0.5,
    backend: str = "auto",
    method: str = "lkm",
    model_id: str = "hustvl/vitmatte-small-composition-1k",
    device: str = "cuda",
    use_fp16: bool = True,
    use_roi: bool = True,
    roi_margin: int = 32,
    max_dim: int = 1600,
    post_blur: float = 0.0,
) -> np.ndarray:
    """
    Refine mask alpha with trimap-based natural image matting.
    Falls back to the input mask on any failure.
    """
    if guide_image is None:
        return _normalize_mask(mask)

    binary = _as_binary(mask, threshold=threshold)
    if not np.any(binary):
        return np.zeros_like(mask, dtype=np.float32)
    if np.all(binary):
        return np.ones_like(mask, dtype=np.float32)

    trimap = _build_trimap(binary, radius=trimap_radius)
    unknown_count = int((trimap == 0.5).sum())
    if unknown_count == 0:
        return binary.astype(np.float32)

    image = np.clip(guide_image.astype(np.float32, copy=False), 0.0, 1.0)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    work_image, work_trimap = image, trimap
    roi = (0, 0, trimap.shape[1], trimap.shape[0])
    if use_roi:
        work_image, work_trimap, roi = _crop_to_unknown_roi(
            image, trimap, margin=roi_margin
        )

    backend_name = str(backend or "auto").strip().lower()
    if backend_name not in {"auto", "vitmatte", "cpu"}:
        backend_name = "auto"

    alpha_work = None
    if backend_name in {"auto", "vitmatte"}:
        try:
            alpha_work = _estimate_alpha_vitmatte(
                image=work_image,
                trimap=work_trimap,
                model_id=model_id,
                device=device,
                use_fp16=use_fp16,
                max_dim=max_dim,
            )
        except Exception:
            alpha_work = None

    if alpha_work is None:
        try:
            alpha_work = _estimate_alpha_pymatting(
                image=work_image,
                trimap=work_trimap,
                method=method,
                max_dim=max_dim,
            )
        except Exception:
            return _normalize_mask(mask)

    # Start from known trimap constraints and inject predicted unknown band.
    alpha = np.where(trimap >= 1.0, 1.0, np.where(trimap <= 0.0, 0.0, mask)).astype(
        np.float32
    )
    x1, y1, x2, y2 = roi
    unknown_work = work_trimap == 0.5
    region = alpha[y1:y2, x1:x2]
    region[unknown_work] = alpha_work[unknown_work]
    alpha[y1:y2, x1:x2] = region
    alpha = _normalize_mask(alpha)

    if post_blur > 0:
        alpha = gaussian_filter(alpha, sigma=float(post_blur))

    # Keep trimap constraints fixed after blur as well.
    alpha[trimap <= 0.0] = 0.0
    alpha[trimap >= 1.0] = 1.0
    return _normalize_mask(alpha)


def refine_guided_edge(mask: np.ndarray, guide_image: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Edge-aware mask refinement using guided filtering.
    Restricts updates to a narrow boundary band and preserves binary support.
    """
    if guide_image is None or radius <= 0:
        return _normalize_mask(mask)

    import cv2

    base = _normalize_mask(mask)
    binary = _as_binary(base)

    guide = guide_image.astype(np.float32, copy=False)
    if guide.ndim == 2:
        guide = np.clip(guide, 0.0, 1.0)
    else:
        guide = np.clip(guide, 0.0, 1.0)

    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            refined = cv2.ximgproc.guidedFilter(
                guide=guide,
                src=base,
                radius=int(radius),
                eps=float(eps),
                dDepth=-1,
            )
        else:
            # Fallback if ximgproc is unavailable.
            refined = cv2.bilateralFilter(base, d=-1, sigmaColor=0.08, sigmaSpace=float(radius))
    except Exception:
        return base

    refined = _normalize_mask(refined)

    # Apply guided result only around edges to avoid changing interior uniformity.
    band = max(1, int(radius // 2))
    dist_in = distance_transform_edt(binary)
    dist_out = distance_transform_edt(~binary)
    dist_to_boundary = np.where(binary, dist_in, dist_out)
    boundary_band = dist_to_boundary <= band

    blended = base.copy()
    blended[boundary_band] = refined[boundary_band]

    # Keep outside at 0 for hard support masks to prevent halo.
    blended = np.where(binary, blended, 0.0)
    return _normalize_mask(blended)


def refine_invert(mask: np.ndarray) -> np.ndarray:
    """Invert mask: selected becomes unselected and vice versa."""
    return 1.0 - mask


def composite_intersect(masks: list) -> np.ndarray:
    """Intersect multiple masks (element-wise minimum)."""
    result = masks[0].copy()
    for m in masks[1:]:
        result = np.minimum(result, m)
    return result


def composite_add(masks: list) -> np.ndarray:
    """Add multiple masks (clamped union)."""
    result = np.zeros_like(masks[0], dtype=np.float32)
    for m in masks:
        result = result + m
    return np.clip(result, 0.0, 1.0)


def apply_refinements(mask: np.ndarray, refine: dict, guide_image: np.ndarray = None) -> np.ndarray:
    """Apply a chain of refinement operations specified in the refine dict."""
    mask = _normalize_mask(mask)
    if not refine:
        return mask

    # Optional pre-contraction to reduce boundary spill.
    contract_px = int(refine.get("contract", 0))
    if contract_px > 0:
        mask = refine_contract(mask, pixels=contract_px)

    # Optional trimap-based matting for semantic boundaries.
    matting_applied = False
    if refine.get("matting_refine", False) and guide_image is not None:
        mask = refine_trimap_matting(
            mask,
            guide_image=guide_image,
            trimap_radius=int(refine.get("trimap_radius", 6)),
            threshold=float(refine.get("trimap_threshold", 0.5)),
            backend=str(refine.get("matting_backend", "auto")),
            method=str(refine.get("matting_method", "lkm")),
            model_id=str(
                refine.get(
                    "matting_model_id", "hustvl/vitmatte-small-composition-1k"
                )
            ),
            device=str(refine.get("matting_device", "cuda")),
            use_fp16=bool(refine.get("matting_fp16", True)),
            use_roi=bool(refine.get("matting_use_roi", True)),
            roi_margin=int(refine.get("matting_roi_margin", 32)),
            max_dim=int(refine.get("matting_max_dim", 1600)),
            post_blur=float(refine.get("matting_post_blur", 0.0)),
        )
        matting_applied = True

    feather_mode = str(refine.get("feather_mode", "edge_inward")).strip().lower()

    # Feathering mode: "edge_inward" (default), "edge" or "gaussian".
    if "feather" in refine and (
        not matting_applied or refine.get("feather_after_matting", False)
    ):
        sigma = float(refine["feather"])
        if feather_mode == "gaussian":
            mask = refine_feather_gaussian(mask, sigma=sigma)
        elif feather_mode == "edge":
            mask = refine_feather_edge(mask, sigma=sigma)
        else:
            mask = refine_feather_edge_inward(mask, sigma=sigma)

    # Optional guided edge snapping.
    if refine.get("guided_refine", False) and guide_image is not None:
        guided_radius = int(refine.get("guided_radius", 8))
        guided_eps = float(refine.get("guided_eps", 1e-3))
        mask = refine_guided_edge(mask, guide_image=guide_image, radius=guided_radius, eps=guided_eps)

    # Invert
    if refine.get("invert", False):
        mask = refine_invert(mask)

    return _normalize_mask(mask)
