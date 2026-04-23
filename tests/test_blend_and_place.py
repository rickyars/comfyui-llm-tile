import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from utils.image_utils import blend_and_place_tile


def _canvas(h=2048, w=2048):
    return torch.zeros(1, h, w, 3)


def test_no_controlnet_hard_places_full_tile():
    canvas = _canvas()
    tile = torch.ones(1024, 1024, 3) * 0.5
    blend_and_place_tile(canvas, tile, pos_x=1024, pos_y=0,
                         tile_width=1024, tile_height=1024,
                         overlap_x=154, overlap_y=0,
                         has_left=True, has_top=False,
                         controlnet_active=False)
    assert canvas[0, 0:1024, 1024:2048, :].mean().item() == pytest.approx(0.5)
    assert canvas[0, 0:1024, 0:1024, :].sum().item() == 0.0


def test_extraction_skips_left_overlap():
    """start_x = overlap_x when has_left and controlnet_active."""
    canvas = _canvas()
    gen_tile = torch.zeros(1024, 1024 + 154, 3)
    gen_tile[:, 154:, :] = 0.8
    blend_and_place_tile(canvas, gen_tile, pos_x=1024, pos_y=0,
                         tile_width=1024, tile_height=1024,
                         overlap_x=154, overlap_y=0,
                         has_left=True, has_top=False,
                         controlnet_active=True)
    assert canvas[0, 0:1024, 1024:2048, :].mean().item() == pytest.approx(0.8)


def test_extraction_skips_top_overlap():
    """start_y = overlap_y when has_top and controlnet_active."""
    canvas = _canvas()
    gen_tile = torch.zeros(1024 + 154, 1024, 3)
    gen_tile[154:, :, :] = 0.7
    blend_and_place_tile(canvas, gen_tile, pos_x=0, pos_y=1024,
                         tile_width=1024, tile_height=1024,
                         overlap_x=0, overlap_y=154,
                         has_left=False, has_top=True,
                         controlnet_active=True)
    assert canvas[0, 1024:2048, 0:1024, :].mean().item() == pytest.approx(0.7)


def test_left_seam_blend_crossfades_into_neighbor_zone():
    """Blend writes into canvas[pos_x-overlap_x:pos_x], not into the new tile zone."""
    canvas = _canvas()
    canvas[0, :, 870:1024, :] = 1.0       # tile1's right edge (white)
    gen_tile = torch.ones(1024, 1178, 3)   # overlap zone matches neighbor (white)
    gen_tile[:, 154:, :] = 0.5            # new content is grey
    blend_and_place_tile(canvas, gen_tile, pos_x=1024, pos_y=0,
                         tile_width=1024, tile_height=1024,
                         overlap_x=154, overlap_y=0,
                         has_left=True, has_top=False,
                         controlnet_active=True)
    # blend(1.0, 1.0) = 1.0 — neighbor zone unchanged when match is perfect
    assert canvas[0, 0:1024, 870:1024, :].mean().item() == pytest.approx(1.0, abs=1e-4)
    assert canvas[0, 0:1024, 1024:2048, :].mean().item() == pytest.approx(0.5, abs=1e-4)


def test_first_tile_no_neighbors():
    """First tile (no neighbors) places full tile at origin regardless of controlnet_active."""
    canvas = _canvas()
    tile = torch.ones(1024, 1024, 3) * 0.3
    blend_and_place_tile(canvas, tile, pos_x=0, pos_y=0,
                         tile_width=1024, tile_height=1024,
                         overlap_x=154, overlap_y=154,
                         has_left=False, has_top=False,
                         controlnet_active=True)
    assert canvas[0, 0:1024, 0:1024, :].mean().item() == pytest.approx(0.3)
