import torch
import pytest
from node_detailer_adaptive import _tile_complexity, _otsu_threshold, _tile_otsu_scores, _build_otsu_map


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


def test_tile_quadtree_density_flat_canvas_returns_single_leaf():
    # Uniform canvas: root detail = 0 <= threshold → root stays as 1 leaf
    canvas = torch.zeros(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_quadtree_density(canvas, coords)
    # 1 leaf / (16*16) pixels
    assert result[0] == pytest.approx(1 / (16 * 16))


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
