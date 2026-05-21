import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

# Support both relative imports (in package) and direct imports (in tests)
if __package__:
    from .utils import feather_blend_latent, _compute_center_grid, _compute_tile_coords
else:
    from utils import feather_blend_latent, _compute_center_grid, _compute_tile_coords


def _tile_complexity(canvas, tile_coords):
    """
    canvas: [B, C, H, W] latent tensor
    tile_coords: list of (y1, x1, y2, x2) in latent space
    Returns: list of float — mean absolute gradient magnitude per tile.

    Uses gradient magnitude rather than variance so that edges (face contours,
    hair, object boundaries) register as complex even when the interior is smooth.
    Flat uniform regions (dark backgrounds, plain walls) return near zero.
    """
    result = []
    for (y1, x1, y2, x2) in tile_coords:
        tile = canvas[:, :, y1:y2, x1:x2]
        if tile.shape[2] <= 1 or tile.shape[3] <= 1:
            result.append(0.0)
            continue

        dx = (tile[:, :, :-1, 1:] - tile[:, :, :-1, :-1]).abs()
        dy = (tile[:, :, 1:, :-1] - tile[:, :, :-1, :-1]).abs()
        grad = (dx + dy).mean(dim=1).flatten()

        k = max(1, int(grad.numel() * 0.10))
        result.append(grad.topk(k).values.mean().item())
    return result


def _scores_to_denoise(scores, curve, denoise_min, denoise_max):
    """
    scores: list of float complexity values (one per tile)
    curve: gamma exponent; >1 biases most tiles toward denoise_min
    Returns: list of (t, denoise) tuples where
      t      — pre-curve normalized score in [0,1] (used for heatmap)
      denoise — final per-tile denoise value
    """
    if not scores:
        return []

    v_min = min(scores)
    v_max = max(scores)
    result = []
    for v in scores:
        if v_max == v_min:
            # Equal and flat means there is no evidence that any region deserves
            # the high-denoise path. Equal but non-flat still benefits from it.
            t = 0.0 if v_max <= _QUIET_SCORE_EPSILON else 1.0
        else:
            t = (v - v_min) / (v_max - v_min)
        t_curved = t ** curve
        denoise = denoise_min + t_curved * (denoise_max - denoise_min)
        result.append((t, denoise))
    return result


try:
    from matplotlib import cm as _mpl_cm
    _viridis_fn = _mpl_cm.viridis
except ImportError:
    _viridis_fn = None

_VIRIDIS_STOPS = [
    (0.267, 0.005, 0.329),  # 0.00  dark purple
    (0.254, 0.266, 0.530),  # 0.25  blue
    (0.129, 0.566, 0.551),  # 0.50  teal
    (0.369, 0.789, 0.383),  # 0.75  green
    (0.993, 0.906, 0.144),  # 1.00  yellow
]

_QUIET_SCORE_EPSILON = 1e-8


def _t_to_rgb(t):
    """
    Map t∈[0,1] to RGB using the viridis colormap.
    t=0 → dark purple, t=0.5 → teal, t=1 → yellow.
    Uses matplotlib if available, otherwise interpolates built-in control points.
    """
    if _viridis_fn is not None:
        r, g, b, _ = _viridis_fn(float(t))
        return r, g, b
    t = max(0.0, min(1.0, t))
    n = len(_VIRIDIS_STOPS) - 1
    lo = min(int(t * n), n - 1)
    f = t * n - lo
    r0, g0, b0 = _VIRIDIS_STOPS[lo]
    r1, g1, b1 = _VIRIDIS_STOPS[lo + 1]
    return r0 + f * (r1 - r0), g0 + f * (g1 - g0), b0 + f * (b1 - b0)


def _build_denoise_map(tile_coords, t_values, canvas_h, canvas_w, cols, rows):
    """
    tile_coords: list of (y1, x1, y2, x2) in latent space — exact sampler positions.
    t_values:    list of pre-curve normalized score [0,1], one per tile
    canvas_h, canvas_w: latent-space dimensions (pixel dims = these x 8)
    cols, rows: strides in each axis; grid is (cols+1) x (rows+1) tiles
    Returns: IMAGE tensor [1, canvas_h*8, canvas_w*8, 3]

    Paints each tile from its x1/y1 to the next tile's x1/y1 (or canvas edge
    for the last column/row). The heatmap maps exactly to the sampler grid.
    """
    H_px, W_px = canvas_h * 8, canvas_w * 8
    img = torch.zeros(1, H_px, W_px, 3)

    for (y1, x1, y2, x2), t in zip(tile_coords, t_values):
        px0 = x1 * 8
        px1 = x2 * 8
        py0 = y1 * 8
        py1 = y2 * 8
        r, g, b = _t_to_rgb(t)
        img[0, py0:py1, px0:px1, 0] = r
        img[0, py0:py1, px0:px1, 1] = g
        img[0, py0:py1, px0:px1, 2] = b

        img[0, py0:min(py0 + 2, py1), px0:px1, :] = 1.0
        img[0, max(py1 - 2, py0):py1, px0:px1, :] = 1.0
        img[0, py0:py1, px0:min(px0 + 2, px1), :] = 1.0
        img[0, py0:py1, max(px1 - 2, px0):px1, :] = 1.0

    return img


class LLMAdaptiveTileDetailer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "upscaled_latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise_min": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_max": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "curve": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("refined_latent", "denoise_map")
    FUNCTION = "detail"
    CATEGORY = "image/generation"

    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler,
               denoise_min, denoise_max, curve, tile_size, overlap):

        canvas = upscaled_latent["samples"].clone()
        _, _, H, W = canvas.shape

        tile_l = tile_size // 8
        overlap_l = overlap // 8
        if overlap_l >= tile_l:
            overlap_l = tile_l // 2
            print(f"[LLMAdaptiveTileDetailer] Warning: overlap clamped to "
                  f"{overlap_l * 8}px (overlap must be < tile_size)")

        cols, rows = _compute_center_grid(W, H, tile_l, overlap_l)
        stride = tile_l - overlap_l

        print(f"[LLMAdaptiveTileDetailer] Latent {W}x{H} | "
              f"tile_l={tile_l} overlap_l={overlap_l} stride={stride} | "
              f"grid cols={cols} rows={rows} ({(rows+1)*(cols+1)} tiles)")

        # --- Pass 1: collect valid tile coords and measure complexity ---
        tile_coords = _compute_tile_coords(W, H, tile_l, cols, rows)
        x_starts = sorted({x1 for _, x1, _, _ in tile_coords})
        y_starts = sorted({y1 for y1, _, _, _ in tile_coords})
        print(f"[LLMAdaptiveTileDetailer] grid starts px "
              f"x={[x * 8 for x in x_starts]} y={[y * 8 for y in y_starts]}")

        scores = _tile_complexity(canvas, tile_coords)
        td_pairs = _scores_to_denoise(scores, curve, denoise_min, denoise_max)
        denoise_map_img = _build_denoise_map(tile_coords, [t for t, _ in td_pairs], H, W, cols, rows)

        # --- Pass 2: sample each tile with its computed denoise ---
        pbar = ProgressBar(len(tile_coords))
        n_cols = cols + 1
        for tile_idx, (y1, x1, y2, x2) in enumerate(tile_coords):
            r = tile_idx // n_cols
            c = tile_idx % n_cols
            t_val, tile_denoise = td_pairs[tile_idx]
            score = scores[tile_idx]

            print(f"[LLMAdaptiveTileDetailer] tile ({r},{c}) "
                  f"grad={score:.4f} t={t_val:.2f} denoise={tile_denoise:.3f}")

            tile_seed = seed + tile_idx
            tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

            noise = comfy.sample.prepare_noise(tile_latent, tile_seed, None)
            refined = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler,
                positive, negative, tile_latent,
                denoise=tile_denoise,
            )

            feather_blend_latent(
                canvas, refined, y1, x1, overlap_l,
                has_left=(c > 0 and x1 < tile_coords[tile_idx - 1][3]),
                has_top=(r > 0 and y1 < tile_coords[tile_idx - n_cols][2]),
            )

            comfy.model_management.soft_empty_cache()
            pbar.update(1)

        return ({"samples": canvas}, denoise_map_img)


NODE_CLASS_MAPPINGS = {
    "LLMAdaptiveTileDetailer": LLMAdaptiveTileDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMAdaptiveTileDetailer": "Adaptive Tiled Image Detailer",
}
