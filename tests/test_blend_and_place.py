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


def test_left_seam_blend_uses_alpha_ramp():
    """Blend zone goes from canvas content (alpha=0) to matched zone (alpha=1)."""
    canvas = _canvas()
    canvas[0, :, 870:1024, :] = 0.0   # tile1's right edge is black
    gen_tile = torch.zeros(1024, 1178, 3)
    gen_tile[:, :154, :] = 1.0         # overlap zone is white (perfect ControlNet match)
    gen_tile[:, 154:, :] = 0.5         # new content is grey
    blend_and_place_tile(canvas, gen_tile, pos_x=1024, pos_y=0,
                         tile_width=1024, tile_height=1024,
                         overlap_x=154, overlap_y=0,
                         has_left=True, has_top=False,
                         controlnet_active=True)
    # At start of blend zone (x=870, alpha≈0): should be close to canvas value (0.0)
    assert canvas[0, 0:1024, 870, :].mean().item() == pytest.approx(0.0, abs=0.02)
    # At end of blend zone (x=1023, alpha≈1): should be close to matched value (1.0)
    assert canvas[0, 0:1024, 1023, :].mean().item() == pytest.approx(1.0, abs=0.02)
    # New content zone unchanged
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


def test_top_seam_blend_uses_alpha_ramp():
    """Top blend zone goes from canvas content (alpha=0) to matched zone (alpha=1)."""
    canvas = _canvas()
    canvas[0, 870:1024, :, :] = 0.0   # tile1's bottom edge is black
    gen_tile = torch.zeros(1178, 1024, 3)
    gen_tile[:154, :, :] = 1.0         # top overlap zone is white
    gen_tile[154:, :, :] = 0.5         # new content is grey
    blend_and_place_tile(canvas, gen_tile, pos_x=0, pos_y=1024,
                         tile_width=1024, tile_height=1024,
                         overlap_x=0, overlap_y=154,
                         has_left=False, has_top=True,
                         controlnet_active=True)
    # Start of blend zone (y=870, alpha≈0): close to canvas value (0.0)
    assert canvas[0, 870, 0:1024, :].mean().item() == pytest.approx(0.0, abs=0.02)
    # End of blend zone (y=1023, alpha≈1): close to matched value (1.0)
    assert canvas[0, 1023, 0:1024, :].mean().item() == pytest.approx(1.0, abs=0.02)
    # New content placed correctly
    assert canvas[0, 1024:2048, 0:1024, :].mean().item() == pytest.approx(0.5, abs=1e-4)


def test_corner_blend_fires_with_both_neighbors():
    """Corner zone blends from canvas corner into tile's matched corner."""
    canvas = _canvas()
    canvas[0, 870:1024, 870:1024, :] = 0.0   # corner zone is black
    gen_tile = torch.zeros(1178, 1178, 3)
    gen_tile[:154, :154, :] = 1.0             # matched corner is white
    gen_tile[154:, 154:, :] = 0.5             # new content
    blend_and_place_tile(canvas, gen_tile, pos_x=1024, pos_y=1024,
                         tile_width=1024, tile_height=1024,
                         overlap_x=154, overlap_y=154,
                         has_left=True, has_top=True,
                         controlnet_active=True)
    # Corner start (y=870, x=870): alpha=0, should still be black (0.0)
    assert canvas[0, 870, 870, :].mean().item() == pytest.approx(0.0, abs=0.02)
    # Corner end diagonal (y=1023, x=1023): alpha=1, should be white (1.0)
    assert canvas[0, 1023, 1023, :].mean().item() == pytest.approx(1.0, abs=0.02)
    # New content
    assert canvas[0, 1024:2048, 1024:2048, :].mean().item() == pytest.approx(0.5, abs=1e-4)
