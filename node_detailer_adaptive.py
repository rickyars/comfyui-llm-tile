import heapq
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


def _otsu_threshold(values, bins=256):
    """
    Return an Otsu threshold for a 1D tensor normalized to [0, 1].
    """
    values = values.flatten().float().clamp(0.0, 1.0)
    if values.numel() == 0:
        return 0.0

    hist = torch.histc(values, bins=bins, min=0.0, max=1.0)
    total = hist.sum()
    if total <= 0:
        return 0.0

    centers = torch.linspace(0.0, 1.0, bins, device=hist.device)
    weight_bg = torch.cumsum(hist, dim=0)
    weight_fg = total - weight_bg
    sum_bg = torch.cumsum(hist * centers, dim=0)
    sum_total = sum_bg[-1]

    valid = (weight_bg > 0) & (weight_fg > 0)
    variance = torch.zeros_like(hist)
    mean_bg = sum_bg[valid] / weight_bg[valid]
    mean_fg = (sum_total - sum_bg[valid]) / weight_fg[valid]
    variance[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg - mean_fg) ** 2

    return centers[int(torch.argmax(variance).item())].item()


def _tile_otsu_scores(canvas, tile_coords):
    """
    Score tiles by coverage of the bright class from a global Otsu split.

    This is a no-mask subject/background proxy. It builds a latent intensity map,
    thresholds it globally with Otsu, and returns bright-class coverage per tile.
    """
    intensity = canvas.mean(dim=1, keepdim=True)
    v_min = intensity.min()
    v_max = intensity.max()
    if (v_max - v_min).abs().item() <= _QUIET_SCORE_EPSILON:
        return [0.0 for _ in tile_coords]

    intensity = (intensity - v_min) / (v_max - v_min)
    threshold = _otsu_threshold(intensity)

    foreground = intensity > threshold

    result = []
    for (y1, x1, y2, x2) in tile_coords:
        result.append(foreground[:, :, y1:y2, x1:x2].float().mean().item())
    return result


def _build_otsu_map(canvas):
    """
    Build a pixel-space IMAGE preview of the global Otsu bright-class mask.
    """
    _, _, H, W = canvas.shape
    intensity = canvas.mean(dim=1, keepdim=True)
    v_min = intensity.min()
    v_max = intensity.max()
    if (v_max - v_min).abs().item() <= _QUIET_SCORE_EPSILON:
        mask = torch.zeros_like(intensity)
    else:
        intensity = (intensity - v_min) / (v_max - v_min)
        threshold = _otsu_threshold(intensity)
        mask = (intensity > threshold).float()

    img = mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
    return img.repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)


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


def _smooth_scores(scores, n_rows, n_cols, own_weight=0.7):
    """
    Blend each tile's score with the average of its 4-connected neighbors.
    own_weight: fraction of the tile's own score to retain (rest comes from neighbors).
    """
    neighbor_weight = 1.0 - own_weight
    smoothed = []
    for idx, score in enumerate(scores):
        r, c = divmod(idx, n_cols)
        neighbors = []
        if r > 0:
            neighbors.append(scores[(r - 1) * n_cols + c])
        if r < n_rows - 1:
            neighbors.append(scores[(r + 1) * n_cols + c])
        if c > 0:
            neighbors.append(scores[r * n_cols + (c - 1)])
        if c < n_cols - 1:
            neighbors.append(scores[r * n_cols + (c + 1)])
        neighbor_avg = sum(neighbors) / len(neighbors) if neighbors else score
        smoothed.append(own_weight * score + neighbor_weight * neighbor_avg)
    return smoothed


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


def _tile_quadtree_density(canvas, tile_coords, min_cell=4, detail_threshold=0.01):
    """
    Score tiles by quadtree leaf density in latent space.

    Greedily subdivides each tile, always splitting the most heterogeneous node
    (highest sum of per-channel std devs). Stops when every remaining node either
    has detail <= detail_threshold (uniform enough) or would produce children
    smaller than min_cell in the split direction.

    Score = leaf_count / tile_area. Comparable across tiles of different sizes.
    Higher score → more fine-grained detail in the tile.

    Ported from the greedy quadtree in E:/projects/pixelator/studio/quadtree-builder.js,
    adapted to operate on PyTorch latent tensors instead of Canvas ImageData.
    """
    sample = canvas[0]  # [C, H, W] — batch dim is always 1 in tiled workflows

    def _region_detail(ry, rx, rh, rw):
        if rh * rw < 2:
            return 0.0
        region = sample[:, ry:ry + rh, rx:rx + rw]  # [C, rH, rW]
        return region.std(dim=[1, 2]).sum().item()

    result = []
    for (y1, x1, y2, x2) in tile_coords:
        th = y2 - y1
        tw = x2 - x1
        if th <= 0 or tw <= 0:
            result.append(0.0)
            continue

        # heap entries: (-detail, ry, rx, rh, rw)
        # Python's heapq is a min-heap; negate detail to get max-heap behaviour.
        heap = [(-_region_detail(y1, x1, th, tw), y1, x1, th, tw)]
        leaf_count = 1

        while heap:
            neg_d, ry, rx, rh, rw = heapq.heappop(heap)
            d = -neg_d

            if d <= detail_threshold:
                continue  # uniform enough — stays as a leaf

            half_h = rh // 2
            half_w = rw // 2
            can_h = half_h >= min_cell
            can_w = half_w >= min_cell

            if not can_h and not can_w:
                continue  # too small to split — stays as a leaf

            if can_h and can_w:
                children = [
                    (ry,           rx,           half_h,        half_w),
                    (ry,           rx + half_w,  half_h,        rw - half_w),
                    (ry + half_h,  rx,           rh - half_h,   half_w),
                    (ry + half_h,  rx + half_w,  rh - half_h,   rw - half_w),
                ]
            elif can_h:
                children = [
                    (ry,           rx,  half_h,      rw),
                    (ry + half_h,  rx,  rh - half_h, rw),
                ]
            else:  # can_w only
                children = [
                    (ry,  rx,           rh,  half_w),
                    (ry,  rx + half_w,  rh,  rw - half_w),
                ]

            leaf_count += len(children) - 1  # parent replaced by children
            for cy, cx, ch, cw in children:
                heapq.heappush(heap, (-_region_detail(cy, cx, ch, cw), cy, cx, ch, cw))

        result.append(leaf_count / (th * tw))
    return result


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
                "scoring_method": (["otsu_threshold", "gradient_magnitude", "quadtree_density"], {"default": "otsu_threshold"}),
                "denoise_min": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_max": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "curve": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.01}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "crop_to_tiles": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("refined_latent", "denoise_map", "scoring_map")
    FUNCTION = "detail"
    CATEGORY = "image/generation"

    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler,
               scoring_method, denoise_min, denoise_max, curve, tile_size, overlap, crop_to_tiles):

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
        tile_coords = _compute_tile_coords(W, H, tile_l, cols, rows, overlap_l)
        x_starts = sorted({x1 for _, x1, _, _ in tile_coords})
        y_starts = sorted({y1 for y1, _, _, _ in tile_coords})
        print(f"[LLMAdaptiveTileDetailer] grid starts px "
              f"x={[x * 8 for x in x_starts]} y={[y * 8 for y in y_starts]}")

        if scoring_method == "otsu_threshold":
            scores = _tile_otsu_scores(canvas, tile_coords)
            scoring_map_img = _build_otsu_map(canvas)
        elif scoring_method == "gradient_magnitude":
            scores = _tile_complexity(canvas, tile_coords)
            scoring_map_img = None
        elif scoring_method == "quadtree_density":
            scores = _tile_quadtree_density(canvas, tile_coords)
            scoring_map_img = None
        else:
            raise ValueError(f"Unknown scoring_method: {scoring_method!r}")

        scores = _smooth_scores(scores, rows + 1, cols + 1)
        td_pairs = _scores_to_denoise(scores, curve, denoise_min, denoise_max)
        denoise_map_img = _build_denoise_map(tile_coords, [t for t, _ in td_pairs], H, W, cols, rows)
        if scoring_map_img is None:
            scoring_map_img = denoise_map_img

        # --- Pass 2: sample each tile with its computed denoise ---
        pbar = ProgressBar(len(tile_coords))
        n_cols = cols + 1
        for tile_idx, (y1, x1, y2, x2) in enumerate(tile_coords):
            r = tile_idx // n_cols
            c = tile_idx % n_cols
            t_val, tile_denoise = td_pairs[tile_idx]
            score = scores[tile_idx]

            print(f"[LLMAdaptiveTileDetailer] tile ({r},{c}) "
                  f"{scoring_method}={score:.4f} t={t_val:.2f} denoise={tile_denoise:.3f}")

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

        if crop_to_tiles:
            y1_c, x1_c = tile_coords[0][0], tile_coords[0][1]
            y2_c, x2_c = tile_coords[-1][2], tile_coords[cols][3]
            canvas = canvas[:, :, y1_c:y2_c, x1_c:x2_c]
            denoise_map_img = denoise_map_img[:, y1_c * 8:y2_c * 8, x1_c * 8:x2_c * 8, :]
            scoring_map_img = scoring_map_img[:, y1_c * 8:y2_c * 8, x1_c * 8:x2_c * 8, :]

        return ({"samples": canvas}, denoise_map_img, scoring_map_img)


NODE_CLASS_MAPPINGS = {
    "LLMAdaptiveTileDetailer": LLMAdaptiveTileDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMAdaptiveTileDetailer": "Adaptive Tiled Image Detailer",
}
