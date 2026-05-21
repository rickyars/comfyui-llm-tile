import torch
import pytest
from node_detailer_adaptive import _tile_complexity


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


from node_detailer_adaptive import _scores_to_denoise


def test_uniform_scores_all_return_denoise_max():
    scores = [0.5, 0.5, 0.5]
    result = _scores_to_denoise(scores, curve=1.5, denoise_min=0.05, denoise_max=0.35)
    for t, denoise in result:
        assert t == pytest.approx(1.0)
        assert denoise == pytest.approx(0.35)


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


def test_single_score_returns_denoise_max():
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
    assert result[0, 0, 0].max().item() < 0.50  # viridis(0) is dark


def test_build_denoise_map_bright_for_t_one():
    coords = [(0, 0, 4, 4)]
    t_values = [1.0]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4, cols=0, rows=0)
    assert result[0, 0, 0, 0].item() > 0.90  # viridis(1) yellow: high R
    assert result[0, 0, 0, 1].item() > 0.80  # high G


def test_build_denoise_map_matches_sampler_grid():
    # 2-row, 1-column grid (cols=0, rows=1).
    # Tile positions: r=0 at y1=0, r=1 at y1=2 (both latent).
    # Heatmap should paint: top tile from y=0 to y=2*8=16px, bottom from 16 to 32px.
    coords = [(0, 0, 4, 4), (2, 0, 6, 4)]  # (y1, x1, y2, x2) in latent
    t_values = [0.0, 1.0]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4, cols=0, rows=1)
    # y_starts = [0, 2] latent; top cell py0=0, py1=16; bottom py0=16, py1=32
    top_max = result[0, 8, 16].max().item()    # inside top cell (y=8 < 16)
    bottom_r = result[0, 24, 16, 0].item()    # inside bottom cell (y=24 >= 16)
    assert top_max < 0.50    # dark purple
    assert bottom_r > 0.90   # yellow (high red)
