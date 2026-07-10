import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from utils.image_utils import feather_blend_latent, compute_tile_coords


# ---------------------------------------------------------------------------
# compute_tile_coords: full-coverage clamped grid (MultiDiffusion-style)
# ---------------------------------------------------------------------------

def _coverage(coords, H, W):
    covered = torch.zeros(H, W, dtype=torch.bool)
    for (y1, x1, y2, x2) in coords:
        covered[y1:y2, x1:x2] = True
    return covered


def test_aligned_no_overlap_gives_adjacent_tiles():
    coords, n_cols, n_rows = compute_tile_coords(W=256, H=256, tile_l=128, overlap_l=0)
    assert (n_cols, n_rows) == (2, 2)
    assert coords[0] == (0, 0, 128, 128)
    assert coords[1] == (0, 128, 128, 256)
    assert coords[2] == (128, 0, 256, 128)


def test_overlap_advances_by_stride():
    coords, n_cols, n_rows = compute_tile_coords(W=288, H=128, tile_l=128, overlap_l=32)
    # stride 96: starts 0, 96, clamped last 160
    assert n_rows == 1
    xs = [x1 for (_, x1, _, _) in coords]
    assert xs == [0, 96, 160]
    for (y1, x1, y2, x2) in coords:
        assert x2 - x1 == 128 and y2 - y1 == 128


def test_grid_covers_full_canvas_when_not_tile_aligned():
    # The whole point of the clamped grid: no uncovered margin, ever.
    for W, H in [(300, 200), (129, 128), (255, 257), (40, 44)]:
        coords, _, _ = compute_tile_coords(W=W, H=H, tile_l=128, overlap_l=32)
        assert _coverage(coords, H, W).all(), (W, H)


def test_last_tile_clamps_to_canvas_edge():
    coords, _, _ = compute_tile_coords(W=300, H=128, tile_l=128, overlap_l=0)
    assert coords[-1][3] == 300
    assert coords[-1][1] == 300 - 128  # clamped, not past the edge


def test_canvas_smaller_than_tile_yields_single_canvas_sized_tile():
    coords, n_cols, n_rows = compute_tile_coords(W=40, H=44, tile_l=128, overlap_l=8)
    assert (n_cols, n_rows) == (1, 1)
    assert coords == [(0, 0, 44, 40)]


def test_all_tiles_uniform_size():
    coords, _, _ = compute_tile_coords(W=500, H=300, tile_l=128, overlap_l=16)
    sizes = {(y2 - y1, x2 - x1) for (y1, x1, y2, x2) in coords}
    assert len(sizes) == 1


def test_row_major_grid_shape_consistent():
    coords, n_cols, n_rows = compute_tile_coords(W=300, H=300, tile_l=128, overlap_l=32)
    assert len(coords) == n_cols * n_rows
    # row-major: within a row, y1 constant and x1 increasing
    for r in range(n_rows):
        row = coords[r * n_cols:(r + 1) * n_cols]
        assert len({y1 for (y1, _, _, _) in row}) == 1
        xs = [x1 for (_, x1, _, _) in row]
        assert xs == sorted(xs)


def test_adjacent_tiles_overlap_at_least_overlap_l():
    coords, n_cols, n_rows = compute_tile_coords(W=333, H=222, tile_l=64, overlap_l=16)
    for r in range(n_rows):
        row = coords[r * n_cols:(r + 1) * n_cols]
        for (_, x1a, _, x2a), (_, x1b, _, x2b) in zip(row, row[1:]):
            assert x2a - x1b >= 16


# ---------------------------------------------------------------------------
# feather_blend_latent (shared blending, unchanged behavior)
# ---------------------------------------------------------------------------

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
