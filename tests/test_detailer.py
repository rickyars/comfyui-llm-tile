import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from utils.image_utils import feather_blend_latent, _compute_center_grid, _compute_tile_coords


def test_feather_blend_latent_left_edge():
    overlap_l = 4
    canvas = torch.zeros(1, 4, 8, 16)
    canvas[:, :, :, 0:8] = 1.0

    refined = torch.full((1, 4, 8, 8), 0.5)

    feather_blend_latent(canvas, refined, y1=0, x1=4, overlap_l=overlap_l,
                         has_left=True, has_top=False)

    assert canvas[0, 0, 0, 4].item() == pytest.approx(1.0, abs=1e-5)
    assert canvas[0, 0, 0, 7].item() == pytest.approx(0.5, abs=1e-5)
    assert canvas[0, 0, 0, 8].item() == pytest.approx(0.5, abs=1e-5)
    vals = [canvas[0, 0, 0, 4 + i].item() for i in range(4)]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


def test_feather_blend_latent_top_edge():
    overlap_l = 4
    canvas = torch.zeros(1, 4, 16, 8)
    canvas[:, :, 0:8, :] = 1.0

    refined = torch.full((1, 4, 8, 8), 0.5)

    feather_blend_latent(canvas, refined, y1=4, x1=0, overlap_l=overlap_l,
                         has_left=False, has_top=True)

    assert canvas[0, 0, 4, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert canvas[0, 0, 7, 0].item() == pytest.approx(0.5, abs=1e-5)
    assert canvas[0, 0, 8, 0].item() == pytest.approx(0.5, abs=1e-5)
    vals = [canvas[0, 0, 4 + i, 0].item() for i in range(4)]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


def test_no_overlap_gives_adjacent_tiles():
    cols, rows = _compute_center_grid(W=256, H=256, tile_l=128, overlap_l=0)
    coords = _compute_tile_coords(W=256, H=256, tile_l=128, cols=cols, rows=rows, overlap_l=0)

    assert cols == 1
    assert rows == 1
    assert len(coords) == 4
    # Tiles are adjacent: x2 of tile 0 == x1 of tile 1
    assert coords[0] == (0, 0, 128, 128)
    assert coords[1] == (0, 128, 128, 256)


def test_overlap_creates_positional_tile_overlap():
    # Grid count is unchanged from no-overlap; tiles grow into neighbor territory
    cols, rows = _compute_center_grid(W=256, H=256, tile_l=128, overlap_l=32)
    coords = _compute_tile_coords(W=256, H=256, tile_l=128, cols=cols, rows=rows, overlap_l=32)

    assert cols == 1
    assert rows == 1
    assert len(coords) == 4  # same grid count as no-overlap

    n_cols = cols + 1
    for tile_idx, (y1, x1, y2, x2) in enumerate(coords):
        r, c = divmod(tile_idx, n_cols)
        if c > 0:
            prev_x2 = coords[tile_idx - 1][3]
            assert x1 < prev_x2, "adjacent tiles must overlap positionally"
        if r > 0:
            prev_y2 = coords[tile_idx - n_cols][2]
            assert y1 < prev_y2, "adjacent tiles must overlap positionally"


def test_grid_covers_full_canvas():
    # Use tile-aligned dimensions (standard upscaler use case)
    cols, rows = _compute_center_grid(W=256, H=384, tile_l=128, overlap_l=32)
    coords = _compute_tile_coords(W=256, H=384, tile_l=128, cols=cols, rows=rows, overlap_l=32)

    assert coords[0][0] == 0   # y1 of first tile
    assert coords[0][1] == 0   # x1 of first tile
    assert coords[-1][2] == 384  # y2 of last tile
    assert coords[-1][3] == 256  # x2 of last tile


def test_feather_blend_latent_corner():
    overlap_l = 4
    canvas = torch.zeros(1, 4, 16, 16)
    canvas[:, :, 0:8, 0:8] = 1.0

    refined = torch.full((1, 4, 8, 8), 0.5)

    feather_blend_latent(canvas, refined, y1=4, x1=4, overlap_l=overlap_l,
                         has_left=True, has_top=True)

    assert canvas[0, 0, 4, 4].item() == pytest.approx(1.0, abs=1e-5)
    assert canvas[0, 0, 7, 7].item() == pytest.approx(0.5, abs=1e-5)
    assert canvas[0, 0, 8, 8].item() == pytest.approx(0.5, abs=1e-5)
