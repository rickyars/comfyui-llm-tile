import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from utils.image_utils import feather_blend_latent, _compute_center_grid


def test_feather_blend_latent_left_edge():
    overlap_l = 4
    # Previously placed tile occupies x=0..7 with value 1.0; rest 0.0
    canvas = torch.zeros(1, 4, 8, 16)
    canvas[:, :, :, 0:8] = 1.0

    # New tile placed at x1=4; tile covers x=4..11 in canvas; value 0.5
    refined = torch.full((1, 4, 8, 8), 0.5)

    feather_blend_latent(canvas, refined, y1=0, x1=4, overlap_l=overlap_l,
                         has_left=True, has_top=False)

    # Left edge of overlap (x=4): alpha=0 → old canvas value preserved
    assert canvas[0, 0, 0, 4].item() == pytest.approx(1.0, abs=1e-5)
    # Right edge of overlap (x=7): alpha=1 → full refined value
    assert canvas[0, 0, 0, 7].item() == pytest.approx(0.5, abs=1e-5)
    # Interior beyond overlap (x=8): hard-set to refined
    assert canvas[0, 0, 0, 8].item() == pytest.approx(0.5, abs=1e-5)
    # Ramp is monotonically non-increasing across the overlap zone
    vals = [canvas[0, 0, 0, 4 + i].item() for i in range(4)]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


def test_feather_blend_latent_top_edge():
    overlap_l = 4
    # Previously placed tile occupies y=0..7 with value 1.0
    canvas = torch.zeros(1, 4, 16, 8)
    canvas[:, :, 0:8, :] = 1.0

    # New tile placed at y1=4; tile covers y=4..11; value 0.5
    refined = torch.full((1, 4, 8, 8), 0.5)

    feather_blend_latent(canvas, refined, y1=4, x1=0, overlap_l=overlap_l,
                         has_left=False, has_top=True)

    # Top edge of overlap (y=4): alpha=0 → old canvas value preserved
    assert canvas[0, 0, 4, 0].item() == pytest.approx(1.0, abs=1e-5)
    # Bottom edge of overlap (y=7): alpha=1 → full refined value
    assert canvas[0, 0, 7, 0].item() == pytest.approx(0.5, abs=1e-5)
    # Interior beyond overlap (y=8): hard-set to refined
    assert canvas[0, 0, 8, 0].item() == pytest.approx(0.5, abs=1e-5)
    # Ramp is monotonically non-increasing across the overlap zone
    vals = [canvas[0, 0, 4 + i, 0].item() for i in range(4)]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


def test_center_anchored_grid():
    # 1000x2048 portrait: latent W=125, H=256
    # tile_size=1024 → tile_l=128; overlap=64 → overlap_l=8
    cols, rows, start_x, start_y = _compute_center_grid(W=125, H=256, tile_l=128, overlap_l=8)

    assert cols == 1
    assert rows == 2
    # Image (125) is narrower than tile_l (128); centering goes slightly negative
    assert start_x == -2
    # 4 latent pixels of top/bottom padding: (256 - 248) // 2 = 4
    assert start_y == 4

    # All clamped tile coordinates must stay within latent bounds
    stride = 128 - 8  # 120
    for r in range(rows + 1):
        for c in range(cols + 1):
            y1 = max(0, start_y + r * stride)
            x1 = max(0, start_x + c * stride)
            y2 = min(256, y1 + 128)
            x2 = min(125, x1 + 128)
            assert 0 <= y1 <= 256
            assert 0 <= x1 <= 125
            assert y1 <= y2 <= 256
            assert x1 <= x2 <= 125


def test_feather_blend_latent_corner():
    overlap_l = 4
    # Simulate a 2x2 tile grid: top-left tile placed, now placing bottom-right at y1=4, x1=4
    canvas = torch.zeros(1, 4, 16, 16)
    canvas[:, :, 0:8, 0:8] = 1.0  # top-left tile region

    refined = torch.full((1, 4, 8, 8), 0.5)

    feather_blend_latent(canvas, refined, y1=4, x1=4, overlap_l=overlap_l,
                         has_left=True, has_top=True)

    # Corner point (y=4, x=4): both alpha_x=0 and alpha_y=0 → min=0 → old canvas value
    assert canvas[0, 0, 4, 4].item() == pytest.approx(1.0, abs=1e-5)
    # Diagonal point (y=7, x=7): both alpha_x=1 and alpha_y=1 → min=1 → full refined value
    assert canvas[0, 0, 7, 7].item() == pytest.approx(0.5, abs=1e-5)
    # Interior (y=8, x=8): hard-set to refined
    assert canvas[0, 0, 8, 8].item() == pytest.approx(0.5, abs=1e-5)
