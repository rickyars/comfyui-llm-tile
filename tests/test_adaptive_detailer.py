import torch
import pytest
from node_detailer_adaptive import _otsu_threshold, _tile_otsu_scores, _build_otsu_map
from node_detailer_adaptive import _build_canvas_quadtree


def test_otsu_threshold_splits_two_value_distribution():
    values = torch.tensor([0.0] * 10 + [1.0] * 10)
    threshold = _otsu_threshold(values, bins=16)
    assert 0.0 <= threshold < 1.0


def test_tile_otsu_scores_prefers_bright_class():
    canvas = torch.zeros(1, 4, 8, 8)
    canvas[:, :, 4:8, 4:8] = 1.0
    coords = [(0, 0, 4, 4), (4, 4, 8, 8)]

    result = _tile_otsu_scores(canvas, coords)

    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(1.0)


def test_tile_otsu_scores_does_not_flip_to_smaller_dark_class():
    canvas = torch.ones(1, 4, 8, 8)
    canvas[:, :, 0:2, 0:2] = 0.0
    coords = [(0, 0, 4, 4), (4, 4, 8, 8)]

    result = _tile_otsu_scores(canvas, coords)

    assert result[0] > 0.0
    assert result[1] == pytest.approx(1.0)


def test_build_otsu_map_returns_pixel_space_image():
    canvas = torch.zeros(1, 4, 2, 2)
    canvas[:, :, 1, 1] = 1.0

    result = _build_otsu_map(canvas)

    assert result.shape == (1, 16, 16, 3)
    assert result[0, 0, 0].mean().item() == pytest.approx(0.0)
    assert result[0, 12, 12].mean().item() == pytest.approx(1.0)


from node_detailer_adaptive import _scores_to_denoise


def test_uniform_scores_all_return_denoise_min():
    scores = [0.0, 0.0, 0.0]
    result = _scores_to_denoise(scores, curve=1.5, denoise_min=0.05, denoise_max=0.35)
    for t, denoise in result:
        assert t == pytest.approx(0.0)
        assert denoise == pytest.approx(0.05)


def test_linear_curve_maps_extremes_correctly():
    scores = [0.0, 1.0]
    result = _scores_to_denoise(scores, curve=1.0, denoise_min=0.05, denoise_max=0.35)
    t0, d0 = result[0]
    t1, d1 = result[1]
    assert t0 == pytest.approx(0.0)
    assert d0 == pytest.approx(0.05)
    assert t1 == pytest.approx(1.0)
    assert d1 == pytest.approx(0.35)


def test_curve_gt_one_biases_midpoint_toward_min():
    scores = [0.0, 0.5, 1.0]
    linear = _scores_to_denoise(scores, curve=1.0, denoise_min=0.0, denoise_max=1.0)
    curved = _scores_to_denoise(scores, curve=2.0, denoise_min=0.0, denoise_max=1.0)
    assert curved[1][1] < linear[1][1]


def test_scores_map_absolutely_not_relatively():
    # The image's own max must NOT be stretched to denoise_max: a 0.7-score
    # tile gets the same denoise whether it is alone or beside a 1.0 tile.
    alone = _scores_to_denoise([0.7], curve=1.5, denoise_min=0.05, denoise_max=0.35)
    beside_sharp = _scores_to_denoise([0.7, 1.0], curve=1.5, denoise_min=0.05, denoise_max=0.35)
    expected = 0.05 + (0.7 ** 1.5) * 0.30
    assert alone[0][1] == pytest.approx(expected)
    assert beside_sharp[0][1] == pytest.approx(expected)
    assert alone[0][1] < 0.35  # relative normalization would have hit the max


def test_scores_clamp_outside_unit_range():
    result = _scores_to_denoise([-0.5, 1.5], curve=1.0, denoise_min=0.1, denoise_max=0.4)
    assert result[0][0] == pytest.approx(0.0)
    assert result[0][1] == pytest.approx(0.1)
    assert result[1][0] == pytest.approx(1.0)
    assert result[1][1] == pytest.approx(0.4)


from node_detailer_adaptive import _t_to_rgb, _build_denoise_map


def test_t_to_rgb_zero_is_dark_purple():
    r, g, b = _t_to_rgb(0.0)
    assert r < 0.40
    assert g < 0.10
    assert b > 0.20


def test_t_to_rgb_one_is_yellow():
    r, g, b = _t_to_rgb(1.0)
    assert r > 0.90
    assert g > 0.80
    assert b < 0.20


def test_t_to_rgb_half_is_teal():
    r, g, b = _t_to_rgb(0.5)
    assert r < 0.25
    assert g > 0.40
    assert b > 0.40


def test_build_denoise_map_shape():
    # Single tile: 1×1 grid
    coords = [(0, 0, 4, 4)]
    t_values = [0.5]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4, n_cols=1, n_rows=1)
    assert result.shape == (1, 32, 32, 3)


def test_build_denoise_map_dark_for_t_zero():
    coords = [(0, 0, 4, 4)]
    t_values = [0.0]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4, n_cols=1, n_rows=1)
    assert result[0, 16, 16].max().item() < 0.50  # viridis(0) is dark purple


def test_build_denoise_map_bright_for_t_one():
    coords = [(0, 0, 4, 4)]
    t_values = [1.0]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4, n_cols=1, n_rows=1)
    assert result[0, 0, 0, 0].item() > 0.90  # viridis(1) yellow: high R
    assert result[0, 0, 0, 1].item() > 0.80  # high G


def test_build_denoise_map_splits_overlap_at_midpoint():
    # Two overlapping tiles in one column: [0,4) and [2,6) overlap on [2,4).
    # Ownership boundary is the overlap midpoint (3), so the map paints
    # [0,3) with the first tile's colour and [3,6) with the second's.
    coords = [(0, 0, 4, 4), (2, 0, 6, 4)]  # (y1, x1, y2, x2) in latent
    t_values = [0.0, 1.0]
    result = _build_denoise_map(coords, t_values, canvas_h=6, canvas_w=4, n_cols=1, n_rows=2)
    assert result[0, 8, 16].max().item() < 0.50   # y=1: first tile, dark purple
    assert result[0, 20, 16, 0].item() < 0.50     # y=2.5: still first tile's side
    assert result[0, 26, 16, 0].item() > 0.90     # y=3.25: second tile, yellow
    assert result[0, 40, 16, 0].item() > 0.90     # y=5: second tile


def test_build_denoise_map_covers_full_canvas_with_clamped_grid():
    # Non-aligned canvas: the clamped grid's last tiles overlap more than
    # overlap_l. The ownership cells must still partition the whole canvas —
    # no unpainted (black) region anywhere.
    from utils.image_utils import compute_tile_coords
    coords, n_cols, n_rows = compute_tile_coords(W=44, H=40, tile_l=16, overlap_l=4)
    t_values = [1.0] * len(coords)  # yellow everywhere
    result = _build_denoise_map(coords, t_values, canvas_h=40, canvas_w=44,
                                n_cols=n_cols, n_rows=n_rows)
    assert result.shape == (1, 320, 352, 3)
    # every pixel painted (viridis(1.0) has red ~0.99)
    assert result[0, :, :, 0].min().item() > 0.90


from node_detailer_adaptive import LLMAdaptiveTileDetailer


def test_scoring_method_enum_excludes_gradient_magnitude():
    # gradient_magnitude produced scores in arbitrary units that only meant
    # anything under per-image min-max normalization; removed with it.
    methods = LLMAdaptiveTileDetailer.INPUT_TYPES()["required"]["scoring_method"][0]
    assert "gradient_magnitude" not in methods


def test_return_names_uses_scoring_map_not_otsu_map():
    assert "scoring_map" in LLMAdaptiveTileDetailer.RETURN_NAMES
    assert "otsu_map" not in LLMAdaptiveTileDetailer.RETURN_NAMES


from node_detailer_adaptive import _tile_structure_energy, _build_structure_map


def _carrier(h=32, w=32, seed=0, amp=1.0):
    """Real VAE latents carry unit-amplitude high-frequency noise regardless
    of pixel content (measured on the Z-Image ae). Synthetic fixtures must
    include it or they test a latent that doesn't exist."""
    torch.manual_seed(seed)
    return amp * torch.randn(1, 4, h, w)


def _checker(h=32, w=32, block=4, amp=2.0):
    """Block checkerboard — structure that survives 4x downsampling, standing
    in for real image detail (edges, faces, texture at visible scale)."""
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    c = (((yy // block + xx // block) % 2).float() * 2.0 - 1.0) * amp
    return c.view(1, 1, h, w).expand(1, 4, h, w).contiguous()


def test_structure_energy_flat_tile_scores_zero():
    canvas = torch.zeros(1, 4, 32, 32)
    result = _tile_structure_energy(canvas, [(0, 0, 16, 16)])
    assert result[0] == pytest.approx(0.0)


def test_structure_energy_carrier_noise_scores_low_structure_scores_high():
    # The core real-latent regression (user's batch bug, verified against
    # actual Z-Image VAE encodes): the high-frequency carrier alone — what a
    # soft region's latent looks like — must score low, while carrier plus
    # visible-scale structure must score high. Raw-gradient and blur-ratio
    # measures could not tell these apart (both ~0.9).
    soft = _carrier(seed=0) + torch.linspace(0, 0.5, 32).view(1, 1, 1, 32)
    detailed = _carrier(seed=1) + _checker()
    coords = [(0, 0, 32, 32)]
    assert _tile_structure_energy(soft, coords)[0] < 0.3
    assert _tile_structure_energy(detailed, coords)[0] > 0.6


def test_structure_energy_score_is_context_independent():
    # Same soft quadrant must score identically alone and next to detail —
    # the score is computed purely from the tile's own pixels.
    soft_alone = _carrier(seed=5)
    soft_beside_detail = soft_alone.clone()
    soft_beside_detail[:, :, 16:32, 16:32] += _checker(16, 16)
    coords = [(0, 0, 16, 16)]
    score_alone = _tile_structure_energy(soft_alone, coords)[0]
    score_beside = _tile_structure_energy(soft_beside_detail, coords)[0]
    assert score_alone == pytest.approx(score_beside)


def test_structure_map_shape_and_range():
    canvas = _carrier(8, 8, seed=2)
    img = _build_structure_map(canvas)
    assert img.shape == (1, 64, 64, 3)
    assert img.min().item() >= 0.0
    assert img.max().item() <= 1.0


def test_scoring_method_enum_includes_structure_energy():
    methods = LLMAdaptiveTileDetailer.INPUT_TYPES()["required"]["scoring_method"][0]
    assert "structure_energy" in methods
    assert "blur_sensitivity" not in methods


from node_detailer_adaptive import _tile_quadtree_density


def test_tile_quadtree_density_flat_canvas_scores_zero():
    # Global tree: flat canvas → 1 leaf spanning the whole canvas.
    # Its center (16,16) is outside the queried tile (0,0,16,16), so score = 0.
    canvas = torch.zeros(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_quadtree_density(canvas, coords)
    assert result[0] == pytest.approx(0.0)


def test_tile_quadtree_density_complex_higher_than_flat():
    canvas_complex = _carrier(seed=42) + _checker()
    canvas_flat = torch.zeros(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    flat_score = _tile_quadtree_density(canvas_flat, coords)
    complex_score = _tile_quadtree_density(canvas_complex, coords)
    assert complex_score[0] > flat_score[0]


def test_tile_quadtree_density_ranks_tiles_correctly():
    # Top-left quadrant: carrier only (soft). Bottom-right: carrier + structure.
    canvas = _carrier(seed=99)
    canvas[:, :, 16:32, 16:32] += _checker(16, 16)
    coords = [(0, 0, 16, 16), (16, 16, 32, 32)]
    result = _tile_quadtree_density(canvas, coords)
    assert result[0] < result[1]


def test_tile_quadtree_density_is_bounded_unit_interval():
    # Structure everywhere subdivides to the min_cell floor → exactly 1.0.
    canvas = _carrier(seed=7) + _checker()
    coords = [(0, 0, 32, 32)]
    result = _tile_quadtree_density(canvas, coords)
    assert result[0] == pytest.approx(1.0)


def test_tile_quadtree_density_partial_detail_scores_high():
    # Real detailed content is heterogeneous — subdivision never reaches the
    # min_cell floor everywhere, so raw density tops out well below 1.0
    # (measured ~0.45 on real detailed latents). The score must be referenced
    # to that achievable ceiling, not the theoretical one: a tile whose left
    # half is dense structure should read as high detail, not mid-scale.
    canvas = _carrier(seed=21)
    canvas[:, :, :, 0:16] += _checker(32, 16)
    coords = [(0, 0, 32, 32)]
    result = _tile_quadtree_density(canvas, coords)
    assert result[0] > 0.9


def test_tile_quadtree_density_carrier_only_canvas_scores_low_everywhere():
    # The batch-processing regression, matched to real latent behavior: a
    # canvas that is soft everywhere (carrier noise, no visible-scale
    # structure) must score low in every tile — not have its least-soft tile
    # promoted, and not have the carrier itself mistaken for detail (the old
    # std-based criterion did exactly that: real latents have std ~1-3
    # everywhere, so every cell subdivided and every tile scored 1.0).
    canvas = _carrier(seed=13)
    coords = [(0, 0, 16, 16), (0, 16, 16, 32), (16, 0, 32, 16), (16, 16, 32, 32)]
    result = _tile_quadtree_density(canvas, coords)
    for score in result:
        assert score < 0.15


def test_tile_quadtree_density_score_is_context_independent():
    # A tile's score must not depend on what else is in the image: the same
    # soft quadrant scores low both alone and next to detail. A single
    # leaf-center of granularity (min_cell^2/tile_area = 0.0625 here) is
    # allowed: neighbouring detail changes the tree partition, not the score.
    # Granularity after referencing to _QUADTREE_REF: 0.0625 / 0.45 ≈ 0.139.
    soft_alone = _carrier(seed=3)
    soft_beside_detail = soft_alone.clone()
    soft_beside_detail[:, :, 16:32, 16:32] += _checker(16, 16)
    coords = [(0, 0, 16, 16)]
    score_alone = _tile_quadtree_density(soft_alone, coords)[0]
    score_beside = _tile_quadtree_density(soft_beside_detail, coords)[0]
    assert score_alone < 0.15
    assert score_beside < 0.15
    assert abs(score_alone - score_beside) <= 0.0625 / 0.45 + 1e-6


def test_scoring_method_enum_includes_quadtree_density():
    methods = LLMAdaptiveTileDetailer.INPUT_TYPES()["required"]["scoring_method"][0]
    assert "quadtree_density" in methods


def test_build_canvas_quadtree_flat_returns_one_leaf():
    # Flat canvas: root detail <= epsilon, stays as the single leaf.
    canvas = torch.zeros(1, 4, 32, 32)
    leaves = _build_canvas_quadtree(canvas)
    assert len(leaves) == 1
    assert leaves[0] == (0, 0, 32, 32)


def test_build_canvas_quadtree_complex_returns_multiple_leaves():
    canvas = _carrier(seed=42) + _checker()
    leaves = _build_canvas_quadtree(canvas)
    assert len(leaves) > 4


def test_build_canvas_quadtree_leaves_partition_canvas():
    # Every latent cell must be covered by exactly one leaf (no gaps, no overlaps).
    canvas = _carrier(seed=42) + _checker()
    leaves = _build_canvas_quadtree(canvas)
    coverage = torch.zeros(32, 32, dtype=torch.int)
    for (ry, rx, rh, rw) in leaves:
        coverage[ry:ry + rh, rx:rx + rw] += 1
    assert (coverage == 1).all()


def test_build_canvas_quadtree_complex_region_gets_more_leaves():
    # Bottom-right quadrant has structure; top-left is carrier-only (soft).
    # Only cells whose structure energy exceeds split_threshold subdivide.
    canvas = _carrier(seed=0)
    canvas[:, :, 16:32, 16:32] += _checker(16, 16)
    leaves = _build_canvas_quadtree(canvas)
    # Count leaves whose origin AND full extent lie within each quadrant
    flat_leaves = [l for l in leaves if l[0] + l[2] <= 16 and l[1] + l[3] <= 16]
    complex_leaves = [l for l in leaves if l[0] >= 16 and l[1] >= 16]
    assert len(complex_leaves) > len(flat_leaves)


from unittest.mock import MagicMock
import comfy.sample
import comfy.samplers


def _make_model_mock():
    model = MagicMock()
    model_sampling = MagicMock()
    model_sampling.sigma_min = 0.03
    model_sampling.sigma_max = 14.6
    model.get_model_object.return_value = model_sampling
    model.load_device = torch.device('cpu')
    model.model_options = {}
    return model


def test_adaptive_eta_scaling_formula():
    # Pure math: verify the per-tile eta formula matches the spec
    denoise_min, denoise_max, eta = 0.05, 0.35, 1.0
    drange = denoise_max - denoise_min

    eta_at_min = eta * (denoise_min - denoise_min) / drange
    eta_at_max = eta * (denoise_max - denoise_min) / drange
    eta_at_mid = eta * ((denoise_min + denoise_max) / 2 - denoise_min) / drange

    assert eta_at_min == pytest.approx(0.0)
    assert eta_at_max == pytest.approx(1.0)
    assert eta_at_mid == pytest.approx(0.5)


def test_adaptive_detail_uses_sample_custom_not_sample():
    comfy.sample.sample_custom.reset_mock()

    node = LLMAdaptiveTileDetailer()
    node.detail(
        model=_make_model_mock(),
        upscaled_latent={"samples": torch.zeros(1, 4, 32, 32)},
        positive=[], negative=[],
        seed=0, steps=20, cfg=7.0,
        sampler_name="euler", scheduler="normal",
        scoring_method="quadtree_density",
        denoise_min=0.05, denoise_max=0.35,
        curve=1.5, tile_size=256, overlap=0,
        noise_type="gaussian", eta_min=0.0, eta_max=1.0,
    )

    assert comfy.sample.sample_custom.call_count > 0


def test_adaptive_detail_eta_varies_across_tiles():
    # High-score tiles get more eta than low-score tiles.
    # Canvas is 32x32 latent; tile_size=128 → tile_l=16 → 2x2 grid (4 tiles).
    # Top-left tile is flat; bottom-right tile has a steep ramp (high gradient).
    # We use scoring_method="otsu_threshold" so scores pick up pixel-level variation.
    captured_etas = []
    original = comfy.samplers.ksampler

    def capturing(name, extra_options=None):
        if extra_options and 'eta' in extra_options:
            captured_etas.append(extra_options['eta'])
        return original(name, extra_options=extra_options)

    comfy.samplers.ksampler = capturing
    try:
        canvas = torch.zeros(1, 4, 32, 32)
        # Bottom-right 16x16 latent block: 4px-block checkerboard — structure
        # that survives the 4x downsample the structure-energy measure uses.
        canvas[:, :, 16:32, 16:32] = _checker(16, 16)
        node = LLMAdaptiveTileDetailer()
        node.detail(
            model=_make_model_mock(),
            upscaled_latent={"samples": canvas},
            positive=[], negative=[],
            seed=0, steps=20, cfg=7.0,
            sampler_name="euler_ancestral", scheduler="normal",
            scoring_method="quadtree_density",
            denoise_min=0.05, denoise_max=0.35,
            curve=1.0, tile_size=128, overlap=0,
            noise_type="gaussian", eta_min=0.0, eta_max=1.0,
        )
    finally:
        comfy.samplers.ksampler = original

    assert len(captured_etas) > 0
    assert max(captured_etas) > min(captured_etas), "eta should vary across tiles"


def test_adaptive_detail_input_types_include_noise_type_and_eta():
    required = LLMAdaptiveTileDetailer.INPUT_TYPES()["required"]
    assert "noise_type" in required
    assert "eta_min" in required
    assert "eta_max" in required
    assert "edge_mode" not in required  # removed: the clamped grid always covers everything
    assert "crop_to_tiles" not in required


# ---------------------------------------------------------------------------
# Task: full-coverage clamped grid (no edge modes)
# ---------------------------------------------------------------------------

def _adaptive_common():
    return dict(
        model=_make_model_mock(), positive=[], negative=[],
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="normal",
        scoring_method="quadtree_density", denoise_min=0.05, denoise_max=0.35,
        curve=1.5, tile_size=128, overlap=0, noise_type="gaussian",
        eta_min=0.0, eta_max=1.0,
    )


def test_adaptive_output_keeps_original_shape_when_not_aligned():
    node = LLMAdaptiveTileDetailer()
    latent = torch.randn(1, 4, 40, 44)
    canvas, denoise_map, scoring_map = node.detail(
        upscaled_latent={"samples": latent.clone()},
        **_adaptive_common(),
    )
    assert canvas["samples"].shape == latent.shape
    # maps are pixel-resolution (x8) and must match the latent size
    assert denoise_map.shape[1] == 40 * 8 and denoise_map.shape[2] == 44 * 8
    assert scoring_map.shape[1] == 40 * 8 and scoring_map.shape[2] == 44 * 8


def test_adaptive_diffuses_entire_canvas_when_not_aligned():
    # The reason edge_mode was removed: every latent pixel must be sampled,
    # even when the canvas is not a multiple of tile_size. Stub sampling to
    # add +100 per tile; with denoise_min > 0 every tile runs, so the whole
    # output must shift — no untouched margins anywhere.
    node = LLMAdaptiveTileDetailer()
    latent = torch.randn(1, 4, 40, 44)

    original = comfy.sample.sample_custom
    comfy.sample.sample_custom = MagicMock(
        side_effect=lambda model, noise, cfg, sampler, sigmas, pos, neg, lat, **kw: lat + 100.0
    )
    try:
        canvas, _, _ = node.detail(
            upscaled_latent={"samples": latent.clone()},
            **_adaptive_common(),
        )
    finally:
        comfy.sample.sample_custom = original

    assert torch.all(canvas["samples"] > latent + 50.0)
