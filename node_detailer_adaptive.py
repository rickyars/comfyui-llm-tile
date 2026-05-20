import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

# Support both relative imports (in package) and direct imports (in tests)
if __package__:
    from .utils import feather_blend_latent, _compute_center_grid
else:
    from utils import feather_blend_latent, _compute_center_grid


def _tile_variances(canvas, tile_coords):
    """
    canvas: [B, C, H, W] latent tensor
    tile_coords: list of (y1, x1, y2, x2) in latent space
    Returns: list of float — per-tile spatial variance averaged across channels
    """
    return [
        canvas[:, :, y1:y2, x1:x2].var(dim=[2, 3]).mean().item()
        for (y1, x1, y2, x2) in tile_coords
    ]


def _variances_to_denoise(variances, curve, denoise_min, denoise_max):
    """
    variances: list of float (one per tile)
    curve: gamma exponent; >1 biases most tiles toward denoise_min
    Returns: list of (t, denoise) tuples where
      t      — pre-curve normalized variance in [0,1] (used for heatmap)
      denoise — final per-tile denoise value
    """
    v_min = min(variances)
    v_max = max(variances)
    result = []
    for v in variances:
        t = 1.0 if v_max == v_min else (v - v_min) / (v_max - v_min)
        t_curved = t ** curve
        denoise = denoise_min + t_curved * (denoise_max - denoise_min)
        result.append((t, denoise))
    return result


def _t_to_rgb(t):
    """
    Map t∈[0,1] to RGB via HSV hue sweep (S=1, V=1).
    t=0 → blue (hue=240°), t=0.5 → green (hue=120°), t=1 → red (hue=0°).
    """
    hue = (1.0 - t) * 240.0   # degrees: 0=red, 120=green, 240=blue
    h = hue / 60.0
    i = int(h) % 6
    f = h - int(h)
    q = 1.0 - f
    segments = [
        (1.0, f,   0.0),  # 0: red→yellow
        (q,   1.0, 0.0),  # 1: yellow→green
        (0.0, 1.0, f  ),  # 2: green→cyan
        (0.0, q,   1.0),  # 3: cyan→blue
        (f,   0.0, 1.0),  # 4: blue→magenta
        (1.0, 0.0, q  ),  # 5: magenta→red
    ]
    return segments[i]


def _build_denoise_map(tile_coords, t_values, canvas_h, canvas_w):
    """
    tile_coords: list of (y1, x1, y2, x2) in latent space
    t_values:    list of pre-curve normalized variance [0,1], one per tile
    canvas_h, canvas_w: latent-space dimensions (pixel dims = these × 8)
    Returns: IMAGE tensor [1, canvas_h*8, canvas_w*8, 3]
    """
    H, W = canvas_h * 8, canvas_w * 8
    img = torch.zeros(1, H, W, 3)
    for (y1, x1, y2, x2), t in zip(tile_coords, t_values):
        r, g, b = _t_to_rgb(t)
        img[0, y1 * 8:y2 * 8, x1 * 8:x2 * 8, 0] = r
        img[0, y1 * 8:y2 * 8, x1 * 8:x2 * 8, 1] = g
        img[0, y1 * 8:y2 * 8, x1 * 8:x2 * 8, 2] = b
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

        cols, rows, start_x, start_y = _compute_center_grid(W, H, tile_l, overlap_l)
        stride = tile_l - overlap_l

        print(f"[LLMAdaptiveTileDetailer] Latent {W}x{H} | "
              f"tile_l={tile_l} overlap_l={overlap_l} stride={stride} | "
              f"grid cols={cols} rows={rows} ({(rows+1)*(cols+1)} tiles)")

        # --- Pass 1: collect valid tile coords and measure variance ---
        tile_coords = []
        for r in range(rows + 1):
            for c in range(cols + 1):
                y1 = max(0, start_y + r * stride)
                x1 = max(0, start_x + c * stride)
                y2 = min(H, y1 + tile_l)
                x2 = min(W, x1 + tile_l)
                if y2 > y1 and x2 > x1:
                    tile_coords.append((y1, x1, y2, x2))

        variances = _tile_variances(canvas, tile_coords)
        td_pairs = _variances_to_denoise(variances, curve, denoise_min, denoise_max)
        denoise_map_img = _build_denoise_map(tile_coords, [t for t, _ in td_pairs], H, W)

        # --- Pass 2: sample each tile with its computed denoise ---
        pbar = ProgressBar(len(tile_coords))
        tile_idx = 0
        for r in range(rows + 1):
            for c in range(cols + 1):
                y1 = max(0, start_y + r * stride)
                x1 = max(0, start_x + c * stride)
                y2 = min(H, y1 + tile_l)
                x2 = min(W, x1 + tile_l)
                if y2 <= y1 or x2 <= x1:
                    continue

                t_val, tile_denoise = td_pairs[tile_idx]
                var = variances[tile_idx]
                tile_idx += 1

                print(f"[LLMAdaptiveTileDetailer] tile ({r},{c}) "
                      f"var={var:.4f} t={t_val:.2f} denoise={tile_denoise:.3f}")

                tile_seed = seed + r * (cols + 1) + c
                tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

                noise = comfy.sample.prepare_noise(tile_latent, tile_seed, None)
                refined = comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler,
                    positive, negative, tile_latent,
                    denoise=tile_denoise,
                )

                feather_blend_latent(
                    canvas, refined, y1, x1, overlap_l,
                    has_left=(x1 > 0),
                    has_top=(y1 > 0),
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
