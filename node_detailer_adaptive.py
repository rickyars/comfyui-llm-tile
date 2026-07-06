import torch
import torch.nn.functional as F
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

if __package__:
    from .utils import (
        feather_blend_latent, _compute_center_grid, _compute_tile_coords,
        check_eta_support, prepare_noise_typed, build_tile_sampler,
        NOISE_GENERATOR_NAMES_SIMPLE, pad_latent_to_grid,
    )
else:
    from utils import (
        feather_blend_latent, _compute_center_grid, _compute_tile_coords,
        check_eta_support, prepare_noise_typed, build_tile_sampler,
        NOISE_GENERATOR_NAMES_SIMPLE, pad_latent_to_grid,
    )


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
    scores: list of float in [0, 1] — absolute per-tile detail scores
    curve: gamma exponent; >1 biases most tiles toward denoise_min
    Returns: list of (t, denoise) tuples where
      t      — the (clamped) input score, used for the heatmap
      denoise — final per-tile denoise value

    The mapping is absolute: a tile's denoise depends only on its own score,
    never on the other tiles in the image. Earlier versions min-max normalized
    scores within the image, which made every image — however soft — stretch
    to denoise_max somewhere; that made batch processing with fixed min/max
    impossible (a soft tile's denoise depended on what it shared the image with).
    Both scoring methods now emit scores that are already meaningful in [0, 1].
    """
    result = []
    for v in scores:
        t = max(0.0, min(1.0, v))
        denoise = denoise_min + (t ** curve) * (denoise_max - denoise_min)
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


def _build_denoise_map(tile_coords, t_values, canvas_h, canvas_w, cols, rows, tile_l):
    """
    tile_coords: list of (y1, x1, y2, x2) in latent space — exact sampler positions.
    t_values:    list of pre-curve normalized score [0,1], one per tile
    canvas_h, canvas_w: latent-space dimensions (pixel dims = these x 8)
    cols, rows: strides in each axis; grid is (cols+1) x (rows+1) tiles
    tile_l: tile edge length in latent units — used to recover the tile anchor.
    Returns: IMAGE tensor [1, canvas_h*8, canvas_w*8, 3]

    Paints each tile on its non-overlapping anchor stride (``[x2 - tile_l, x2)``)
    rather than its full sampled span (``[x1, x2)``). The sampled span extends
    ``overlap_l`` pixels leftward/upward into the previous tile, so painting it
    with a row-major hard overwrite lets the right/lower tile win every shared
    overlap zone — which biases the whole map leftward and, in pad/crop modes,
    collapses the near-edge sliver while inflating the far-edge one. Anchor
    strides tile the canvas exactly (no overlap, no gaps), so the heatmap
    reflects the true centered grid.
    """
    H_px, W_px = canvas_h * 8, canvas_w * 8
    img = torch.zeros(1, H_px, W_px, 3)

    for (y1, x1, y2, x2), t in zip(tile_coords, t_values):
        px0 = (x2 - tile_l) * 8
        px1 = x2 * 8
        py0 = (y2 - tile_l) * 8
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


def _grad_energy(x):
    """Mean squared forward-difference gradient of a [B, C, h, w] region."""
    if x.shape[2] < 2 or x.shape[3] < 2:
        return 0.0
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.pow(2).mean().item() + dy.pow(2).mean().item()


# Gradient-energy floor for blur_sensitivity, in squared-latent-gradient units.
# The bare ratio is amplitude-blind: a near-flat region whose only gradient
# energy is low-amplitude noise (VAE dither, grain) scores ~1 because blurring
# destroys noise just as thoroughly as real detail. Energy near this floor
# collapses the score to 0; energy well above it leaves the ratio unchanged.
# Unit-variance raw noise has energy ~4; observed soft/flat latent regions sit
# around 1e-3 or below.
_BLUR_ENERGY_FLOOR = 0.01


def _tile_blur_sensitivity(canvas, tile_coords):
    """
    Score tiles by how much of their gradient energy a small blur destroys:

        score = 1 - (grad_energy(blurred) + floor) / (grad_energy(tile) + floor)

    Fine detail (texture, sharp edges) is annihilated by a 3x3 blur, so
    detailed tiles score near 1. Smooth content barely changes under blurring,
    and near-flat content has energy at the noise floor — both score near 0.
    The score is an absolute [0, 1] sharpness measure, comparable across
    tiles, images, and batches.
    """
    blurred = F.avg_pool2d(canvas, kernel_size=3, stride=1, padding=1,
                           count_include_pad=False)
    result = []
    for (y1, x1, y2, x2) in tile_coords:
        e = _grad_energy(canvas[:, :, y1:y2, x1:x2])
        eb = _grad_energy(blurred[:, :, y1:y2, x1:x2])
        score = 1.0 - (eb + _BLUR_ENERGY_FLOOR) / (e + _BLUR_ENERGY_FLOOR)
        result.append(max(0.0, score))
    return result


def _build_blur_sensitivity_map(canvas, window=5):
    """
    Pixel-space grayscale preview of local blur sensitivity: white = gradient
    energy that a 3x3 blur would destroy (fine detail), black = smooth/flat.

    Returns: IMAGE tensor [1, H*8, W*8, 3]
    """
    def _sq_grad(x):
        g = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        if x.shape[2] < 2 or x.shape[3] < 2:
            return g
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).mean(dim=1, keepdim=True)
        dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).mean(dim=1, keepdim=True)
        g[:, :, :, :-1] += dx
        g[:, :, :-1, :] += dy
        return g

    canvas = canvas.detach().cpu()
    blurred = F.avg_pool2d(canvas, kernel_size=3, stride=1, padding=1,
                           count_include_pad=False)
    pad = window // 2
    e = F.avg_pool2d(_sq_grad(canvas), window, stride=1, padding=pad,
                     count_include_pad=False)
    eb = F.avg_pool2d(_sq_grad(blurred), window, stride=1, padding=pad,
                      count_include_pad=False)
    mask = (1.0 - (eb + _BLUR_ENERGY_FLOOR) / (e + _BLUR_ENERGY_FLOOR)).clamp(0.0, 1.0)

    img = mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
    return img.repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)


_DEFAULT_SPLIT_THRESHOLD = 0.35


def _region_detail(sample, ry, rx, rh, rw):
    """Mean per-channel std of a region — channel-count invariant, so the same
    split_threshold works for 4-channel (SDXL) and 16-channel (z-image) latents."""
    if rh * rw < 2:
        return 0.0
    return sample[:, ry:ry + rh, rx:rx + rw].std(dim=[1, 2]).mean().item()


def _build_canvas_quadtree(canvas, min_cell=4, split_threshold=_DEFAULT_SPLIT_THRESHOLD):
    """
    Run a threshold-based quadtree over the whole canvas.

    A cell splits only when it holds sufficient information — detail (mean
    per-channel std) above split_threshold — and stops at min_cell. There is
    no subdivision budget and no competition between regions: whether a cell
    splits depends only on its own content, so the resulting leaf structure
    is an absolute measure. A uniformly soft canvas genuinely produces few,
    large leaves everywhere (the earlier greedy-heap design force-spent a
    budget on "the least soft of the soft", manufacturing density on images
    that had no detail at all).

    Recursion is bounded by min_cell, so no iteration cap is needed.

    Returns: list of (ry, rx, rh, rw) leaf cells covering the full canvas.
    """
    # _region_detail (.std().item()) runs once per visited cell; on a GPU
    # tensor each .item() is a device sync, so score on CPU instead.
    sample = canvas[0].detach().cpu()  # [C, H, W]
    _, H, W = sample.shape

    stack = [(0, 0, H, W)]
    leaves = []

    while stack:
        ry, rx, rh, rw = stack.pop()

        half_h = rh // 2
        half_w = rw // 2
        can_h = half_h >= min_cell
        can_w = half_w >= min_cell

        if (not can_h and not can_w) or \
                _region_detail(sample, ry, rx, rh, rw) <= split_threshold:
            leaves.append((ry, rx, rh, rw))
            continue

        if can_h and can_w:
            children = [
                (ry,          rx,           half_h,       half_w),
                (ry,          rx + half_w,  half_h,       rw - half_w),
                (ry + half_h, rx,           rh - half_h,  half_w),
                (ry + half_h, rx + half_w,  rh - half_h,  rw - half_w),
            ]
        elif can_h:
            children = [
                (ry,          rx,  half_h,      rw),
                (ry + half_h, rx,  rh - half_h, rw),
            ]
        else:
            children = [
                (ry, rx,          rh,  half_w),
                (ry, rx + half_w, rh,  rw - half_w),
            ]

        stack.extend(children)

    return leaves


def _tile_quadtree_density(canvas, tile_coords, min_cell=4,
                           split_threshold=_DEFAULT_SPLIT_THRESHOLD):
    """
    Score tiles by quadtree leaf density from a single global canvas quadtree.

    Runs one threshold-based quadtree over the whole canvas (see
    _build_canvas_quadtree), then scores each tile by counting leaves whose
    center falls within it, normalized to an absolute [0, 1] scale:

        score = leaves_with_center_in_tile * min_cell^2 / tile_area

    min_cell^2 / tile_area is the reciprocal of the maximum possible leaf
    count for the tile (every leaf at the min_cell floor), so 1.0 means
    "subdivided to the limit everywhere" and 0.0 means "no detail anywhere".
    The score is comparable across images and batches — a soft tile scores
    low regardless of what else is in the image.
    """
    leaves = _build_canvas_quadtree(canvas, min_cell, split_threshold)
    result = []
    for (y1, x1, y2, x2) in tile_coords:
        th = y2 - y1
        tw = x2 - x1
        if th <= 0 or tw <= 0:
            result.append(0.0)
            continue
        count = sum(
            1 for (ry, rx, rh, rw) in leaves
            if y1 <= ry + rh // 2 < y2 and x1 <= rx + rw // 2 < x2
        )
        result.append(count * (min_cell * min_cell) / (th * tw))
    return result


def _build_quadtree_map(canvas, min_cell=4, split_threshold=_DEFAULT_SPLIT_THRESHOLD):
    """
    Build a pixel-space visualization of the global canvas quadtree.

    Draws white cell outlines on a dark background using the same global tree
    as _tile_quadtree_density. Large cells = flat regions. Small cells = detail.

    Returns: IMAGE tensor [1, H*8, W*8, 3]
    """
    sample = canvas[0]
    _, H, W = sample.shape
    img = torch.zeros(1, H * 8, W * 8, 3)

    for (ry, rx, rh, rw) in _build_canvas_quadtree(canvas, min_cell, split_threshold):
        py0, py1 = ry * 8, (ry + rh) * 8
        px0, px1 = rx * 8, (rx + rw) * 8
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
                "scoring_method": (["otsu_threshold", "quadtree_density", "blur_sensitivity"], {"default": "otsu_threshold"}),
                "denoise_min": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_max": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "curve": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.01}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "edge_mode": (["center", "crop", "pad"], {"default": "center", "tooltip": "center: diffuse centered grid, leave edge margins as the original upscale (original size). crop: crop output to the detailed region. pad: edge-replicate to a full tile grid, diffuse everything, crop back to original size."}),
                "noise_type": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                "eta_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                      "tooltip": "Eta for lowest-denoise tiles. 0 = deterministic ODE. Eta-compatible samplers: euler_ancestral, dpmpp_sde, dpmpp_2s_ancestral, dpmpp_2m_sde, dpmpp_3m_sde, rk_beta."}),
                "eta_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                      "tooltip": "Eta for highest-denoise tiles. Scales linearly from eta_min (at denoise_min) to eta_max (at denoise_max)."}),
            },
            "optional": {
                "split_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 5.0, "step": 0.01,
                                              "tooltip": "quadtree_density only: a cell subdivides while its mean per-channel latent std exceeds this. Lower = more sensitive (more tiles count as detailed). VAE latents are roughly unit-variance, so the default suits most models."}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("refined_latent", "denoise_map", "scoring_map")
    FUNCTION = "detail"
    CATEGORY = "image/generation"

    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler,
               scoring_method, denoise_min, denoise_max, curve,
               tile_size, overlap, edge_mode, noise_type, eta_min, eta_max,
               split_threshold=_DEFAULT_SPLIT_THRESHOLD):

        canvas = upscaled_latent["samples"].clone()
        # Video-latent-format models (e.g. Krea2 with the Wan VAE) hand us a 5D
        # (B, C, T, H, W) latent; everything below operates in 4D image space, so
        # collapse a singleton temporal axis here and restore it before returning.
        temporal_latent = canvas.ndim == 5
        if temporal_latent:
            if canvas.shape[2] != 1:
                raise ValueError(
                    "LLMAdaptiveTileDetailer only supports single-frame latents, "
                    f"got temporal dim T={canvas.shape[2]}.")
            canvas = canvas[:, :, 0]
        _, _, H0, W0 = canvas.shape

        tile_l = tile_size // 8
        pad_top = pad_left = 0
        if edge_mode == "pad":
            canvas, (pad_top, pad_left) = pad_latent_to_grid(canvas, tile_l)
        _, _, H, W = canvas.shape
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

        model_sampling = model.get_model_object("model_sampling")
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)
        eta_supported = check_eta_support(sampler_name)
        if eta_max > 0.0 and not eta_supported:
            print(f"[LLMAdaptiveTileDetailer] Warning: '{sampler_name}' does not support "
                  f"eta; eta_min/eta_max will be ignored. Use an ancestral sampler or 'rk_beta'.")
        drange = denoise_max - denoise_min

        # --- Pass 1: collect valid tile coords and measure complexity ---
        tile_coords = _compute_tile_coords(W, H, tile_l, cols, rows, overlap_l)
        x_starts = sorted({x1 for _, x1, _, _ in tile_coords})
        y_starts = sorted({y1 for y1, _, _, _ in tile_coords})
        print(f"[LLMAdaptiveTileDetailer] grid starts px "
              f"x={[x * 8 for x in x_starts]} y={[y * 8 for y in y_starts]}")

        if scoring_method == "otsu_threshold":
            scores = _tile_otsu_scores(canvas, tile_coords)
            scoring_map_img = _build_otsu_map(canvas)
        elif scoring_method == "quadtree_density":
            scores = _tile_quadtree_density(canvas, tile_coords, split_threshold=split_threshold)
            scoring_map_img = _build_quadtree_map(canvas, split_threshold=split_threshold)
        elif scoring_method == "blur_sensitivity":
            scores = _tile_blur_sensitivity(canvas, tile_coords)
            scoring_map_img = _build_blur_sensitivity_map(canvas)
        else:
            raise ValueError(f"Unknown scoring_method: {scoring_method!r}")

        scores = _smooth_scores(scores, rows + 1, cols + 1)
        td_pairs = _scores_to_denoise(scores, curve, denoise_min, denoise_max)
        denoise_map_img = _build_denoise_map(tile_coords, [t for t, _ in td_pairs], H, W, cols, rows, tile_l)

        # --- Pass 2: sample each tile with its computed denoise and scaled eta ---
        pbar = ProgressBar(len(tile_coords))
        n_cols = cols + 1
        for tile_idx, (y1, x1, y2, x2) in enumerate(tile_coords):
            r = tile_idx // n_cols
            c = tile_idx % n_cols
            t_val, tile_denoise = td_pairs[tile_idx]
            score = scores[tile_idx]

            print(f"[LLMAdaptiveTileDetailer] tile ({r},{c}) "
                  f"{scoring_method}={score:.4f} t={t_val:.2f} denoise={tile_denoise:.3f}")

            if tile_denoise <= 0.0:
                pbar.update(1)
                continue

            t_eta = (tile_denoise - denoise_min) / drange if drange > 0 else 0.0
            tile_eta = eta_min + (eta_max - eta_min) * t_eta
            tile_seed = seed + tile_idx
            tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

            if tile_denoise >= 1.0:
                tile_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
            else:
                new_steps = int(steps / tile_denoise)
                tile_sigmas = comfy.samplers.calculate_sigmas(
                    model_sampling, scheduler, new_steps)[-(steps + 1):]
            tile_sigmas = tile_sigmas.to(model.load_device)

            tile_sampler = build_tile_sampler(sampler_name, tile_eta, eta_supported)
            # Match the model's expected latent rank before sampling. Video-latent
            # models (e.g. Krea2 uses the Wan21 format with latent_dimensions=3)
            # expect a 5D (B, C, T, H, W) latent; feeding a raw 4D image latent and
            # 4D noise lets them broadcast into a phantom temporal axis, which the
            # model then folds into the batch and mismatches the conditioning.
            # fix_empty_latent_channels adds the T=1 axis when the model needs it and
            # is a no-op for ordinary 4D image models. Generate noise from the
            # prepared latent so noise and latent stay the same rank.
            model_tile_latent = comfy.sample.fix_empty_latent_channels(model, tile_latent)
            noise = prepare_noise_typed(model_tile_latent, tile_seed, noise_type, sigma_min, sigma_max)
            refined = comfy.sample.sample_custom(
                model, noise, cfg, tile_sampler, tile_sigmas,
                positive, negative, model_tile_latent,
            )
            if refined.ndim == 5:
                refined = refined.squeeze(2)

            feather_blend_latent(
                canvas, refined, y1, x1, overlap_l,
                has_left=(c > 0 and x1 < tile_coords[tile_idx - 1][3]),
                has_top=(r > 0 and y1 < tile_coords[tile_idx - n_cols][2]),
            )

            # Flushing the CUDA cache every tile costs real time and mostly frees
            # memory the next tile immediately re-allocates; throttle it.
            if (tile_idx + 1) % 4 == 0:
                comfy.model_management.soft_empty_cache()
            pbar.update(1)

        comfy.model_management.soft_empty_cache()

        if edge_mode == "crop":
            y1_c, x1_c = tile_coords[0][0], tile_coords[0][1]
            y2_c, x2_c = tile_coords[-1][2], tile_coords[cols][3]
            canvas = canvas[:, :, y1_c:y2_c, x1_c:x2_c]
            denoise_map_img = denoise_map_img[:, y1_c * 8:y2_c * 8, x1_c * 8:x2_c * 8, :]
            scoring_map_img = scoring_map_img[:, y1_c * 8:y2_c * 8, x1_c * 8:x2_c * 8, :]
        elif edge_mode == "pad":
            canvas = canvas[:, :, pad_top:pad_top + H0, pad_left:pad_left + W0]
            denoise_map_img = denoise_map_img[
                :, pad_top * 8:(pad_top + H0) * 8, pad_left * 8:(pad_left + W0) * 8, :]
            scoring_map_img = scoring_map_img[
                :, pad_top * 8:(pad_top + H0) * 8, pad_left * 8:(pad_left + W0) * 8, :]

        if temporal_latent:
            canvas = canvas.unsqueeze(2)
        return ({"samples": canvas}, denoise_map_img, scoring_map_img)


NODE_CLASS_MAPPINGS = {
    "LLMAdaptiveTileDetailer": LLMAdaptiveTileDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMAdaptiveTileDetailer": "Adaptive Tiled Image Detailer",
}
