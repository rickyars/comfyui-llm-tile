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
