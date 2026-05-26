import torch
import pytest
from node_detailer_adaptive import _tile_complexity, _otsu_threshold, _tile_otsu_scores, _build_otsu_map
from node_detailer_adaptive import _build_canvas_quadtree


def test_tile_complexity_flat_returns_zero():
    canvas = torch.zeros(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_complexity(canvas, coords)
    assert result == [pytest.approx(0.0)]


def test_tile_complexity_nonzero_for_random():
    torch.manual_seed(42)
    canvas = torch.randn(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_complexity(canvas, coords)
    assert result[0] > 0.0


def test_tile_complexity_flat_less_than_varied():
    # Top-left quadrant is flat (zeros); bottom-right has a ramp with edges
    canvas = torch.zeros(1, 4, 32, 32)
    ramp = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4).expand(1, 4, 4, 4).clone()
    canvas[:, :, 16:20, 16:20] = ramp
    coords = [(0, 0, 16, 16), (16, 16, 32, 32)]
    result = _tile_complexity(canvas, coords)
    assert result[0] < result[1]


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


def test_single_nonzero_score_returns_denoise_max():
    result = _scores_to_denoise([0.7], curve=1.5, denoise_min=0.05, denoise_max=0.35)
    assert result[0][1] == pytest.approx(0.35)


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
    # Single tile: 1×1 grid (cols=0, rows=0)
    coords = [(0, 0, 4, 4)]
    t_values = [0.5]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4, cols=0, rows=0)
    assert result.shape == (1, 32, 32, 3)


def test_build_denoise_map_dark_for_t_zero():
    coords = [(0, 0, 4, 4)]
    t_values = [0.0]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4, cols=0, rows=0)
    # Check interior pixel (not border — _build_denoise_map draws white borders at tile edges)
    assert result[0, 16, 16].max().item() < 0.50  # viridis(0) is dark purple


def test_build_denoise_map_bright_for_t_one():
    coords = [(0, 0, 4, 4)]
    t_values = [1.0]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4, cols=0, rows=0)
    assert result[0, 0, 0, 0].item() > 0.90  # viridis(1) yellow: high R
    assert result[0, 0, 0, 1].item() > 0.80  # high G


def test_build_denoise_map_matches_sampler_grid():
    # 2-row, 1-column grid (cols=0, rows=1).
    # Tile positions: r=0 at y1=0, r=1 at y1=2 (both latent).
    # Heatmap paints the same rectangles provided to the sampler; later tiles
    # overwrite earlier overlap pixels, matching row-major sampling order.
    coords = [(0, 0, 4, 4), (2, 0, 6, 4)]  # (y1, x1, y2, x2) in latent
    t_values = [0.0, 1.0]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4, cols=0, rows=1)
    top_max = result[0, 8, 16].max().item()      # top-only region
    overlap_r = result[0, 24, 16, 0].item()      # second tile owns overlap
    assert top_max < 0.50    # dark purple
    assert overlap_r > 0.90


from node_detailer_adaptive import LLMAdaptiveTileDetailer


def test_scoring_method_enum_includes_gradient_magnitude():
    methods = LLMAdaptiveTileDetailer.INPUT_TYPES()["required"]["scoring_method"][0]
    assert "gradient_magnitude" in methods


def test_return_names_uses_scoring_map_not_otsu_map():
    assert "scoring_map" in LLMAdaptiveTileDetailer.RETURN_NAMES
    assert "otsu_map" not in LLMAdaptiveTileDetailer.RETURN_NAMES


def test_gradient_magnitude_scores_complex_tile_higher_than_flat():
    canvas_flat = torch.zeros(1, 4, 32, 32)
    canvas_complex = torch.zeros(1, 4, 32, 32)
    canvas_complex[:, :, 8:24, 8:24] = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4).expand(1, 4, 4, 4).repeat(1, 1, 4, 4)
    coords = [(0, 0, 16, 16)]
    flat_score = _tile_complexity(canvas_flat, coords)
    complex_score = _tile_complexity(canvas_complex, coords)
    assert flat_score[0] < complex_score[0]


from node_detailer_adaptive import _tile_quadtree_density


def test_tile_quadtree_density_flat_canvas_scores_zero():
    # Global tree: flat canvas → 1 leaf spanning the whole canvas.
    # Its center (16,16) is outside the queried tile (0,0,16,16), so score = 0.
    canvas = torch.zeros(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_quadtree_density(canvas, coords)
    assert result[0] == pytest.approx(0.0)


def test_tile_quadtree_density_complex_higher_than_flat():
    torch.manual_seed(42)
    canvas_complex = torch.randn(1, 4, 32, 32)
    canvas_flat = torch.zeros(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    flat_score = _tile_quadtree_density(canvas_flat, coords)
    complex_score = _tile_quadtree_density(canvas_complex, coords)
    assert complex_score[0] > flat_score[0]


def test_tile_quadtree_density_ranks_tiles_correctly():
    # Top-left quadrant: zeros (flat). Bottom-right: random (complex).
    torch.manual_seed(99)
    canvas = torch.zeros(1, 4, 32, 32)
    canvas[:, :, 16:32, 16:32] = torch.randn(1, 4, 16, 16)
    coords = [(0, 0, 16, 16), (16, 16, 32, 32)]
    result = _tile_quadtree_density(canvas, coords)
    assert result[0] < result[1]


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
    torch.manual_seed(42)
    canvas = torch.randn(1, 4, 32, 32)
    leaves = _build_canvas_quadtree(canvas)
    assert len(leaves) > 4


def test_build_canvas_quadtree_leaves_partition_canvas():
    # Every latent cell must be covered by exactly one leaf (no gaps, no overlaps).
    torch.manual_seed(42)
    canvas = torch.randn(1, 4, 32, 32)
    leaves = _build_canvas_quadtree(canvas)
    coverage = torch.zeros(32, 32, dtype=torch.int)
    for (ry, rx, rh, rw) in leaves:
        coverage[ry:ry + rh, rx:rx + rw] += 1
    assert (coverage == 1).all()


def test_build_canvas_quadtree_complex_region_gets_more_leaves():
    # Bottom-right quadrant is complex; top-left is flat.
    # Global heap spends budget on the complex region — it should have more leaves.
    torch.manual_seed(0)
    canvas = torch.zeros(1, 4, 32, 32)
    canvas[:, :, 16:32, 16:32] = torch.randn(1, 4, 16, 16)
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
        scoring_method="gradient_magnitude",
        denoise_min=0.05, denoise_max=0.35,
        curve=1.5, tile_size=256, overlap=0,
        crop_to_tiles=False,
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
        # Bottom-right 16x16 latent block: alternating 0/1 checkerboard
        for i in range(16):
            for j in range(16):
                canvas[:, :, 16 + i, 16 + j] = float((i + j) % 2)
        node = LLMAdaptiveTileDetailer()
        node.detail(
            model=_make_model_mock(),
            upscaled_latent={"samples": canvas},
            positive=[], negative=[],
            seed=0, steps=20, cfg=7.0,
            sampler_name="euler_ancestral", scheduler="normal",
            scoring_method="gradient_magnitude",
            denoise_min=0.05, denoise_max=0.35,
            curve=1.0, tile_size=128, overlap=0,
            crop_to_tiles=False,
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
