import torch
import pytest
from node_detailer_adaptive import _tile_variances


def test_tile_variances_flat_returns_zero():
    canvas = torch.zeros(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_variances(canvas, coords)
    assert result == [pytest.approx(0.0)]


def test_tile_variances_nonzero_for_random():
    torch.manual_seed(42)
    canvas = torch.randn(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_variances(canvas, coords)
    assert result[0] > 0.0


def test_tile_variances_flat_less_than_varied():
    # Top-left quadrant is flat (zeros); bottom-right has a ramp (non-zero variance)
    canvas = torch.zeros(1, 4, 32, 32)
    ramp = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4).expand(1, 4, 4, 4).clone()
    canvas[:, :, 16:20, 16:20] = ramp
    coords = [(0, 0, 16, 16), (16, 16, 32, 32)]
    result = _tile_variances(canvas, coords)
    assert result[0] < result[1]


from node_detailer_adaptive import _variances_to_denoise


def test_uniform_variances_all_return_denoise_max():
    variances = [0.5, 0.5, 0.5]
    result = _variances_to_denoise(variances, curve=1.5, denoise_min=0.05, denoise_max=0.35)
    for t, denoise in result:
        assert t == pytest.approx(1.0)
        assert denoise == pytest.approx(0.35)


def test_linear_curve_maps_extremes_correctly():
    variances = [0.0, 1.0]
    result = _variances_to_denoise(variances, curve=1.0, denoise_min=0.05, denoise_max=0.35)
    t0, d0 = result[0]
    t1, d1 = result[1]
    assert t0 == pytest.approx(0.0)
    assert d0 == pytest.approx(0.05)   # flat → denoise_min
    assert t1 == pytest.approx(1.0)
    assert d1 == pytest.approx(0.35)   # detailed → denoise_max


def test_curve_gt_one_biases_midpoint_toward_min():
    variances = [0.0, 0.5, 1.0]
    linear = _variances_to_denoise(variances, curve=1.0, denoise_min=0.0, denoise_max=1.0)
    curved = _variances_to_denoise(variances, curve=2.0, denoise_min=0.0, denoise_max=1.0)
    # 0.5^1.0=0.5 vs 0.5^2.0=0.25 → curved midpoint gets lower denoise
    assert curved[1][1] < linear[1][1]


def test_single_variance_returns_denoise_max():
    # Single tile → v_min == v_max → t=1.0 → denoise_max
    result = _variances_to_denoise([0.7], curve=1.5, denoise_min=0.05, denoise_max=0.35)
    assert result[0][1] == pytest.approx(0.35)
