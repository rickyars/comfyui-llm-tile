import torch
import torch.nn.functional as F
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

if __package__:
    from .utils import (
        feather_blend_latent, compute_tile_coords,
        check_eta_support, prepare_noise_typed, build_tile_sampler,
        NOISE_GENERATOR_NAMES_SIMPLE,
    )
else:
    from utils import (
        feather_blend_latent, compute_tile_coords,
        check_eta_support, prepare_noise_typed, build_tile_sampler,
        NOISE_GENERATOR_NAMES_SIMPLE,
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

    This function itself is a pure per-score mapping, but note the scoring
    methods feeding it are canvas-relative (each tile is ranked within its
    own canvas's energy distribution — see _tile_structure_energy and
    _tile_quadtree_density), so scores, and therefore denoise values, are
    NOT comparable across different images: the same physical tile can land
    at different denoise in different canvases. Absolute cross-batch scoring
    was tried and abandoned — no fixed energy constant survives a VAE or
    upscale-ratio change.
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


def _build_denoise_map(tile_coords, t_values, canvas_h, canvas_w, n_cols, n_rows):
    """
    tile_coords: list of (y1, x1, y2, x2) in latent space — exact sampler positions.
    t_values:    list of pre-curve normalized score [0,1], one per tile
    canvas_h, canvas_w: latent-space dimensions (pixel dims = these x 8)
    n_cols, n_rows: grid shape; tile_coords[r * n_cols + c] is row r, column c.
    Returns: IMAGE tensor [1, canvas_h*8, canvas_w*8, 3]

    Paints each tile's ownership cell rather than its full sampled span. Tiles
    overlap (including the clamped last tile in each axis, which can overlap
    its neighbor by more than the configured overlap), so painting full spans
    with a row-major overwrite would bias the map toward later tiles. The
    ownership boundary between two adjacent tiles is the midpoint of their
    shared overlap zone; the cells partition the canvas exactly.
    """
    H_px, W_px = canvas_h * 8, canvas_w * 8
    img = torch.zeros(1, H_px, W_px, 3)

    x_starts = sorted({x1 for (_, x1, _, _) in tile_coords})
    y_starts = sorted({y1 for (y1, _, _, _) in tile_coords})
    tw = tile_coords[0][3] - tile_coords[0][1]
    th = tile_coords[0][2] - tile_coords[0][0]

    def _bounds(starts, length, limit):
        cuts = [0]
        for prev, cur in zip(starts, starts[1:]):
            cuts.append((cur + prev + length) // 2)
        cuts.append(limit)
        return cuts

    xb = _bounds(x_starts, tw, canvas_w)
    yb = _bounds(y_starts, th, canvas_h)

    for idx, t in enumerate(t_values):
        r, c = divmod(idx, n_cols)
        red, green, blue = _t_to_rgb(t)
        cell = img[0, yb[r] * 8:yb[r + 1] * 8, xb[c] * 8:xb[c + 1] * 8]
        cell[:, :, 0] = red
        cell[:, :, 1] = green
        cell[:, :, 2] = blue

    return img


# --- Latent structure energy ------------------------------------------------
# VAE latents carry a high-amplitude, high-frequency "carrier" regardless of
# pixel content — raw latent gradient energy alone can't tell soft regions
# from detailed ones. Downsampling the latent 4x averages the carrier away;
# the gradient energy of what survives is actual image structure.
#
# The *absolute* magnitude of that energy is not portable: it depends on
# which VAE encoded the latent (measured ~2.5x different between the
# Z-Image/Flux ae and the Qwen Image VAE for the same source image) and how
# much the canvas was upscaled before encoding (lanczos upscaling smooths
# real detail, lowering measured energy the more a canvas was stretched).
# Comparing against any fixed constant broke on every VAE/scale change we
# tried. What's actually being asked — "where in *this* image is there more
# detail than elsewhere" — is a per-canvas ranking question, not an absolute
# one, so scoring is done by percentile rank within the canvas's own energy
# distribution: scale- and VAE-invariant by construction, since it never
# looks at absolute magnitude at all.
_STRUCTURE_POOL = 4


def _latent_structure_map(canvas):
    """
    Per-latent-pixel structure-energy map [B, 1, H, W].

    Squared forward-difference gradients of the 4x-downsampled latent
    (channel-mean), upsampled back to latent resolution with nearest so
    regions can be scored at any granularity.
    """
    x = canvas.detach().cpu()
    # avg_pool2d raises when a spatial dim is smaller than the kernel
    # (latent strips under 4px); shrink the kernel rather than crash.
    pool = max(1, min(_STRUCTURE_POOL, x.shape[-2], x.shape[-1]))
    p = F.avg_pool2d(x, pool)
    g = torch.zeros(p.shape[0], 1, p.shape[2], p.shape[3])
    if p.shape[2] >= 2 and p.shape[3] >= 2:
        dx = (p[:, :, :, 1:] - p[:, :, :, :-1]).pow(2).mean(dim=1, keepdim=True)
        dy = (p[:, :, 1:, :] - p[:, :, :-1, :]).pow(2).mean(dim=1, keepdim=True)
        g[:, :, :, :-1] += dx
        g[:, :, :-1, :] += dy
    return F.interpolate(g, size=canvas.shape[-2:], mode="nearest")


_STRUCTURE_PERCENTILE = 0.85  # per-tile: "does this tile contain detail", not "is it detailed on average"
_STRUCTURE_FLOOR_PERCENTILE = 0.05   # per-canvas: tile energy that maps to score 0
_STRUCTURE_CEIL_PERCENTILE = 0.95    # per-canvas: tile energy that maps to score 1
# A p95/p05 *ratio* test was tried to gate out "no real detail anywhere"
# canvases (pure carrier noise) before ranking — scale-invariant in theory,
# since it never looks at absolute magnitude. It doesn't hold up: pure
# synthetic carrier noise measured ratio ~6-15x at pixel granularity, but a
# real photograph's actual tile-to-tile detail separation measured only
# ~2.9x after per-tile percentile aggregation (aggregation smooths the
# extremes further than raw pixels) — lower than the "pure noise" case it
# was meant to reject. There is no statistic here that reliably tells
# "real but subtle detail" apart from "no detail at all"; only an exact
# zero-variance canvas (every tile identical) is unambiguous, so that's all
# that's special-cased below.
_STRUCTURE_DIVIDE_EPS = 1e-9


def _tile_structure_values(canvas, tile_coords):
    """
    Per-tile structure energy: the _STRUCTURE_PERCENTILE-th percentile of
    structure energy within the tile (not the mean — see module docstring
    above _tile_structure_energy for why). Returns raw (unnormalized)
    values, shared by _tile_structure_energy and _build_structure_map so
    both use the same canvas-relative low/high bounds.
    """
    smap = _latent_structure_map(canvas)
    return smap, [
        torch.quantile(smap[:, :, y1:y2, x1:x2].flatten(), _STRUCTURE_PERCENTILE).item()
        for (y1, x1, y2, x2) in tile_coords
    ]


def _canvas_structure_bounds(values):
    """
    (low, high) energy bounds for this canvas's own tile values, at
    _STRUCTURE_FLOOR_PERCENTILE / _STRUCTURE_CEIL_PERCENTILE. Returns
    (0, 0) only for a canvas with exactly zero energy spread (every tile
    identical) — the one case where ranking is meaningless rather than just
    low-contrast.
    """
    t = torch.tensor(values)
    lo = torch.quantile(t, _STRUCTURE_FLOOR_PERCENTILE).item()
    hi = torch.quantile(t, _STRUCTURE_CEIL_PERCENTILE).item()
    if hi - lo < _STRUCTURE_DIVIDE_EPS:
        return 0.0, 0.0
    return lo, hi


def _block_structure_bounds(smap, min_cell=4):
    """
    Canvas-relative (lo, hi) bounds from min_cell-block energies instead of
    per-tile energies. Used when the tile grid is too small to rank against
    itself: with 1 tile, p05/p95 of a single value collapse (lo == hi, every
    tile forced to 0); with 2-3 tiles they pin near-identical tiles to the
    0/1 extremes. Block granularity always gives a real distribution.
    """
    H, W = smap.shape[-2:]
    hb, wb = max(1, H // min_cell), max(1, W // min_cell)
    ch, cw = min(min_cell, H), min(min_cell, W)
    blocks = smap[0, 0, :hb * ch, :wb * cw] \
        .unfold(0, ch, ch).unfold(1, cw, cw).reshape(-1, ch * cw)
    energies = torch.quantile(blocks, _STRUCTURE_PERCENTILE, dim=1)
    return _canvas_structure_bounds(energies.tolist())


def _tile_structure_energy(canvas, tile_coords):
    """
    Score tiles by where their structure energy ranks within this canvas's
    own distribution of tile energies — not against any absolute constant.

        score = clamp((tile_energy - canvas_p05) / (canvas_p95 - canvas_p05), 0, 1)

    This is the actual question being asked ("where in this image is there
    more detail than elsewhere"), and it is scale- and VAE-invariant by
    construction: it never compares against a fixed magnitude, so it can't
    be thrown off by which VAE encoded the latent or how much the canvas
    was upscaled beforehand — both of which change the *absolute* energy
    scale but not the *relative* ranking of tiles within one canvas.
    A canvas with no meaningful energy spread anywhere (flat, or carrier
    noise only) scores every tile 0 rather than promoting its least-flat
    tile to look detailed.
    """
    smap, values = _tile_structure_values(canvas, tile_coords)
    # Fewer than 4 tiles can't produce meaningful p05/p95 bounds from their
    # own values (1 tile: lo==hi, forced to 0; 2-3 tiles: near-identical
    # tiles pinned to the extremes) — rank against block-level energies then.
    if len(values) >= 4:
        lo, hi = _canvas_structure_bounds(values)
    else:
        lo, hi = _block_structure_bounds(smap)
    if hi == lo:  # degenerate zero-spread canvas (see _canvas_structure_bounds)
        return [0.0 for _ in values]
    return [max(0.0, min(1.0, (v - lo) / (hi - lo))) for v in values]


def _build_structure_map(canvas, tile_coords=None):
    """
    Pixel-space grayscale preview of latent structure energy, normalized
    against the canvas's own energy distribution (see _tile_structure_energy):
    white = at/above the canvas's own p95, black = at/below its p05.

    tile_coords is used only to establish the canvas-relative bounds; when
    omitted, a coarse fixed grid over the canvas is used instead so the
    preview is still meaningful on its own.

    Returns: IMAGE tensor [1, H*8, W*8, 3]
    """
    smap = _latent_structure_map(canvas)
    if tile_coords is None:
        _, _, H, W = canvas.shape
        # //8 keeps the grid at tile scale on normal canvases; below 8px on
        # the short side that would degenerate to step=1 (one tile per
        # pixel), so floor it at min_cell-sized blocks instead.
        step = max(4, min(H, W) // 8)
        tile_coords = [
            (y, x, min(y + step, H), min(x + step, W))
            for y in range(0, H, step) for x in range(0, W, step)
        ]
    values = [
        torch.quantile(smap[:, :, y1:y2, x1:x2].flatten(), _STRUCTURE_PERCENTILE).item()
        for (y1, x1, y2, x2) in tile_coords
    ]
    if len(values) >= 4:
        lo, hi = _canvas_structure_bounds(values)
    else:
        lo, hi = _block_structure_bounds(smap)
    if hi == lo:
        mask = torch.zeros_like(smap)
    else:
        mask = ((smap - lo) / (hi - lo)).clamp(0.0, 1.0)
    img = mask.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
    return img.repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)


# split_percentile: a cell subdivides while its own structure energy exceeds
# this percentile of the *whole canvas's* energy distribution. A percentile
# of the canvas's own content, not an absolute energy value — scale- and
# VAE-invariant for the same reason _tile_structure_energy is (see its
# docstring): "detailed" is relative to what else is in this image.
_DEFAULT_SPLIT_PERCENTILE = 0.6

def _quadtree_hot_blocks(canvas, min_cell=4, split_percentile=_DEFAULT_SPLIT_PERCENTILE):
    """
    Boolean [hb, wb] grid of "hot" min_cell blocks: blocks whose
    _STRUCTURE_PERCENTILE energy exceeds the canvas's own split_percentile
    quantile of block energies. Threshold and block statistic are computed
    at the SAME granularity — earlier versions compared block statistics
    against per-pixel (or mixed-size-cell) percentiles, which are on a
    different scale: a large cell's percentile converges to the canvas-wide
    one and always clears a lower-percentile threshold, so even pure-noise
    canvases subdivided fully and every tile saturated to 1.0.

    Structure energy rather than raw latent std is essential: real VAE
    latents have std ~1-3 *everywhere* (soft or detailed), so a std-based
    criterion marks soft images just as heavily as detailed ones.

    Returns (hot, H, W): hot mask over blocks; H, W the latent dims.
    Edges: the canvas is replicate-padded up to a multiple of min_cell so
    right/bottom remainder strips participate in the threshold.
    """
    smap = _latent_structure_map(canvas)[0, 0]  # [H, W] CPU
    H, W = smap.shape
    pad_h = (-H) % min_cell
    pad_w = (-W) % min_cell
    if pad_h or pad_w:
        smap = F.pad(smap[None, None], (0, pad_w, 0, pad_h), mode="replicate")[0, 0]
    hb, wb = smap.shape[0] // min_cell, smap.shape[1] // min_cell
    blocks = smap.unfold(0, min_cell, min_cell).unfold(1, min_cell, min_cell) \
        .reshape(hb, wb, min_cell * min_cell)
    block_energy = torch.quantile(blocks, _STRUCTURE_PERCENTILE, dim=2)
    # Threshold as a VALUE interpolated across the canvas's own p05..p95
    # block-energy range — not a count percentile. A count percentile (p-th
    # quantile of block energies) guarantees (1-p) of blocks are hot on ANY
    # canvas, including pure carrier noise where all blocks are nearly equal
    # — which is exactly how every tile saturated before. With a value
    # threshold, near-equal blocks all sit below it (nothing is meaningfully
    # hotter than the rest), while real detail pockets clear it decisively.
    lo = torch.quantile(block_energy.flatten(), _STRUCTURE_FLOOR_PERCENTILE).item()
    hi = torch.quantile(block_energy.flatten(), _STRUCTURE_CEIL_PERCENTILE).item()
    if hi - lo < _STRUCTURE_DIVIDE_EPS:
        return torch.zeros_like(block_energy, dtype=torch.bool), H, W
    threshold = lo + split_percentile * (hi - lo)
    return block_energy > threshold, H, W


def _build_canvas_quadtree(canvas, min_cell=4, split_percentile=_DEFAULT_SPLIT_PERCENTILE):
    """
    Run a quadtree over the whole canvas on the hot-block grid.

    A cell splits while it contains at least one hot block (see
    _quadtree_hot_blocks) and is larger than one block per axis, isolating
    each pocket of above-threshold detail down to the min_cell floor while
    leaving flat regions as large leaves. Recursion is bounded by the block
    grid, so no iteration cap is needed.

    Returns: list of (ry, rx, rh, rw) latent-coordinate leaf cells that
    partition the full canvas.
    """
    hot, H, W = _quadtree_hot_blocks(canvas, min_cell, split_percentile)
    hb, wb = hot.shape
    # 2D prefix sum for O(1) any-hot-in-rect queries.
    csum = hot.to(torch.int32).cumsum(0).cumsum(1)

    def _hot_count(by, bx, bh, bw):
        total = csum[by + bh - 1, bx + bw - 1].item()
        if by > 0:
            total -= csum[by - 1, bx + bw - 1].item()
        if bx > 0:
            total -= csum[by + bh - 1, bx - 1].item()
        if by > 0 and bx > 0:
            total += csum[by - 1, bx - 1].item()
        return total

    stack = [(0, 0, hb, wb)]
    leaves = []
    while stack:
        by, bx, bh, bw = stack.pop()
        can_h = bh >= 2
        can_w = bw >= 2
        if (not can_h and not can_w) or _hot_count(by, bx, bh, bw) == 0:
            # Convert block coords to latent coords, clipping the padded edge.
            ry, rx = by * min_cell, bx * min_cell
            rh = min(bh * min_cell, H - ry)
            rw = min(bw * min_cell, W - rx)
            if rh > 0 and rw > 0:
                leaves.append((ry, rx, rh, rw))
            continue
        half_h = bh // 2
        half_w = bw // 2
        if can_h and can_w:
            stack += [(by, bx, half_h, half_w),
                      (by, bx + half_w, half_h, bw - half_w),
                      (by + half_h, bx, bh - half_h, half_w),
                      (by + half_h, bx + half_w, bh - half_h, bw - half_w)]
        elif can_h:
            stack += [(by, bx, half_h, bw), (by + half_h, bx, bh - half_h, bw)]
        else:
            stack += [(by, bx, bh, half_w), (by, bx + half_w, bh, bw - half_w)]

    return leaves


def _tile_quadtree_density(canvas, tile_coords, min_cell=4,
                           split_percentile=_DEFAULT_SPLIT_PERCENTILE):
    """
    Score tiles by hot-block coverage: the fraction of the tile's area
    covered by blocks whose energy meaningfully exceeds the canvas's own
    energy range (see _quadtree_hot_blocks). A fully detailed tile is fully
    hot and scores 1.0; a flat tile has no hot blocks and scores 0.0; a
    canvas with no meaningful spread anywhere has no hot blocks at all.
    No fixed ceiling constant — the old _QUADTREE_REF=0.45 was calibrated
    under an absolute threshold and saturated once thresholds became
    canvas-relative. Scores rank tiles within this canvas.
    """
    hot, H, W = _quadtree_hot_blocks(canvas, min_cell, split_percentile)
    # Per-latent-pixel hot mask, clipped back to the unpadded canvas.
    hotpix = hot.repeat_interleave(min_cell, 0).repeat_interleave(min_cell, 1)[:H, :W]
    raws = []
    for (y1, x1, y2, x2) in tile_coords:
        if y2 <= y1 or x2 <= x1:
            raws.append(0.0)
            continue
        raws.append(hotpix[y1:y2, x1:x2].float().mean().item())
    # Raw coverage rarely approaches 1.0 on real content (detail is
    # heterogeneous within a tile — measured max ~0.25 on a real painting at
    # the default percentile), which would compress every tile toward
    # denoise_min. Reference to the canvas's own p95 tile coverage so the
    # busiest tiles reach 1.0 — same canvas-relative normalization as
    # structure_energy. Flat tiles stay exactly 0 (no hot blocks at all).
    if len(raws) >= 4:
        hi = torch.quantile(torch.tensor(raws), _STRUCTURE_CEIL_PERCENTILE).item()
    else:
        # Too few tiles to rank against each other (a single tile would
        # always normalize to 1.0 against itself); raw coverage is already
        # a meaningful in-tile fraction, use it directly.
        hi = 1.0
    if hi < _STRUCTURE_DIVIDE_EPS:
        return [0.0 for _ in raws]
    return [min(1.0, r / hi) for r in raws]


def _build_quadtree_map(canvas, min_cell=4, split_percentile=_DEFAULT_SPLIT_PERCENTILE):
    """
    Build a pixel-space visualization of the global canvas quadtree.

    Draws white cell outlines on a dark background using the same global tree
    as _tile_quadtree_density. Large cells = flat regions. Small cells = detail.

    Returns: IMAGE tensor [1, H*8, W*8, 3]
    """
    sample = canvas[0]
    _, H, W = sample.shape
    img = torch.zeros(1, H * 8, W * 8, 3)

    for (ry, rx, rh, rw) in _build_canvas_quadtree(canvas, min_cell, split_percentile):
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
                "scoring_method": (["otsu_threshold", "quadtree_density", "structure_energy"], {"default": "otsu_threshold"}),
                "denoise_min": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_max": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "curve": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.01}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "noise_type": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                "eta_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                      "tooltip": "Eta for lowest-denoise tiles. 0 = deterministic ODE. Eta-compatible samplers: euler_ancestral, dpmpp_sde, dpmpp_2s_ancestral, dpmpp_2m_sde, dpmpp_3m_sde, rk_beta."}),
                "eta_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                      "tooltip": "Eta for highest-denoise tiles. Scales linearly from eta_min (at denoise_min) to eta_max (at denoise_max)."}),
            },
            "optional": {
                "split_percentile": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
                                               "tooltip": "quadtree_density only: a block counts as detailed when its structure energy exceeds a value this far across the canvas's own low-to-high energy range. Lower = more of the image counts as detailed. Canvas-relative, so it holds steady across VAEs and upscale ratios."}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("refined_latent", "denoise_map", "scoring_map")
    FUNCTION = "detail"
    CATEGORY = "image/generation"

    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler,
               scoring_method, denoise_min, denoise_max, curve,
               tile_size, overlap, noise_type, eta_min, eta_max,
               split_percentile=_DEFAULT_SPLIT_PERCENTILE):

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
        _, _, H, W = canvas.shape

        tile_l = tile_size // 8
        overlap_l = overlap // 8
        if overlap_l >= tile_l:
            overlap_l = tile_l // 2
            print(f"[LLMAdaptiveTileDetailer] Warning: overlap clamped to "
                  f"{overlap_l * 8}px (overlap must be < tile_size)")

        model_sampling = model.get_model_object("model_sampling")
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)
        eta_supported = check_eta_support(sampler_name)
        if eta_max > 0.0 and not eta_supported:
            print(f"[LLMAdaptiveTileDetailer] Warning: '{sampler_name}' does not support "
                  f"eta; eta_min/eta_max will be ignored. Use an ancestral sampler or 'rk_beta'.")
        drange = denoise_max - denoise_min

        # --- Pass 1: build the full-coverage grid and measure complexity ---
        tile_coords, n_cols, n_rows = compute_tile_coords(W, H, tile_l, overlap_l)
        x_starts = sorted({x1 for _, x1, _, _ in tile_coords})
        y_starts = sorted({y1 for y1, _, _, _ in tile_coords})
        print(f"[LLMAdaptiveTileDetailer] Latent {W}x{H} | "
              f"tile_l={tile_l} overlap_l={overlap_l} | "
              f"grid {n_cols}x{n_rows} ({len(tile_coords)} tiles)")
        print(f"[LLMAdaptiveTileDetailer] grid starts px "
              f"x={[x * 8 for x in x_starts]} y={[y * 8 for y in y_starts]}")

        if scoring_method == "otsu_threshold":
            scores = _tile_otsu_scores(canvas, tile_coords)
            scoring_map_img = _build_otsu_map(canvas)
        elif scoring_method == "quadtree_density":
            scores = _tile_quadtree_density(canvas, tile_coords, split_percentile=split_percentile)
            scoring_map_img = _build_quadtree_map(canvas, split_percentile=split_percentile)
        elif scoring_method == "structure_energy":
            scores = _tile_structure_energy(canvas, tile_coords)
            scoring_map_img = _build_structure_map(canvas, tile_coords)
        else:
            raise ValueError(f"Unknown scoring_method: {scoring_method!r}")

        scores = _smooth_scores(scores, n_rows, n_cols)
        td_pairs = _scores_to_denoise(scores, curve, denoise_min, denoise_max)
        denoise_map_img = _build_denoise_map(tile_coords, [t for t, _ in td_pairs], H, W, n_cols, n_rows)

        # --- Pass 2: sample each tile with its computed denoise and scaled eta ---
        pbar = ProgressBar(len(tile_coords))
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
            # Mask to torch's valid seed range: at the widget max
            # (0xffffffffffffffff), seed + tile_idx would overflow and
            # torch.manual_seed raises mid-run.
            tile_seed = (seed + tile_idx) & 0xffffffffffffffff
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

            # Feather across the ACTUAL overlap with the neighbor, not the
            # configured one: the clamped last tile per axis overlaps by
            # more than overlap_l (even when overlap_l == 0), and an
            # unfeathered hard write there leaves a visible seam.
            left_ov = (tile_coords[tile_idx - 1][3] - x1) if c > 0 else 0
            top_ov = (tile_coords[tile_idx - n_cols][2] - y1) if r > 0 else 0
            feather_blend_latent(
                canvas, refined, y1, x1, overlap_l,
                has_left=left_ov > 0, has_top=top_ov > 0,
                overlap_x=max(overlap_l, left_ov), overlap_y=max(overlap_l, top_ov),
            )

            # Flushing the CUDA cache every tile costs real time and mostly frees
            # memory the next tile immediately re-allocates; throttle it.
            if (tile_idx + 1) % 4 == 0:
                comfy.model_management.soft_empty_cache()
            pbar.update(1)

        comfy.model_management.soft_empty_cache()

        if temporal_latent:
            canvas = canvas.unsqueeze(2)
        # Preserve non-samples keys (noise_mask, batch_index) for downstream
        # nodes. Note: noise_mask is passed through, not honored — per-tile
        # sampling here does not mask which regions get re-detailed.
        out_latent = dict(upscaled_latent)
        out_latent["samples"] = canvas
        return (out_latent, denoise_map_img, scoring_map_img)


NODE_CLASS_MAPPINGS = {
    "LLMAdaptiveTileDetailer": LLMAdaptiveTileDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMAdaptiveTileDetailer": "Adaptive Tiled Image Detailer",
}
