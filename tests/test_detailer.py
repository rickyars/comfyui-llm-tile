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


# ---------------------------------------------------------------------------
# Task 4: eta and noise_type inputs
# ---------------------------------------------------------------------------

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


def test_detail_uses_sample_custom_not_sample():
    from node_detailer import LLMTileSequentialDetailer
    comfy.sample.sample_custom.reset_mock()

    node = LLMTileSequentialDetailer()
    node.detail(
        model=_make_model_mock(),
        upscaled_latent={"samples": torch.zeros(1, 4, 32, 32)},
        positive=[], negative=[],
        seed=0, steps=20, cfg=7.0,
        sampler_name="euler", scheduler="normal",
        denoise=0.25, tile_size=256, overlap=0,
        crop_to_tiles=False,
        noise_type="gaussian", eta=0.0,
    )

    assert comfy.sample.sample_custom.call_count > 0


def test_detail_eta_nonzero_passes_eta_to_ksampler():
    from node_detailer import LLMTileSequentialDetailer
    captured = []
    original = comfy.samplers.ksampler

    def capturing(name, extra_options=None):
        captured.append(dict(extra_options) if extra_options else {})
        return original(name, extra_options=extra_options)

    comfy.samplers.ksampler = capturing
    try:
        node = LLMTileSequentialDetailer()
        node.detail(
            model=_make_model_mock(),
            upscaled_latent={"samples": torch.zeros(1, 4, 32, 32)},
            positive=[], negative=[],
            seed=0, steps=20, cfg=7.0,
            sampler_name="euler_ancestral", scheduler="normal",
            denoise=0.25, tile_size=256, overlap=0,
            crop_to_tiles=False,
            noise_type="gaussian", eta=0.8,
        )
    finally:
        comfy.samplers.ksampler = original

    assert any(opts.get('eta') == 0.8 for opts in captured)


def test_detail_input_types_include_noise_type_and_eta():
    from node_detailer import LLMTileSequentialDetailer
    required = LLMTileSequentialDetailer.INPUT_TYPES()["required"]
    assert "noise_type" in required
    assert "eta" in required


# ---------------------------------------------------------------------------
# Task: pad_latent_to_grid
# ---------------------------------------------------------------------------

from utils.image_utils import pad_latent_to_grid


def test_pad_latent_to_grid_pads_to_tile_multiple():
    canvas = torch.zeros(1, 4, 100, 140)  # neither dim a multiple of 64
    padded, (pad_top, pad_left) = pad_latent_to_grid(canvas, tile_l=64)
    assert padded.shape[2] == 128  # ceil(100/64)*64
    assert padded.shape[3] == 192  # ceil(140/64)*64


def test_pad_latent_to_grid_symmetric_split():
    canvas = torch.zeros(1, 4, 100, 140)
    padded, (pad_top, pad_left) = pad_latent_to_grid(canvas, tile_l=64)
    # height pad total = 28 -> top 14, bottom 14
    assert pad_top == 14
    # width pad total = 52 -> left 26, right 26
    assert pad_left == 26


def test_pad_latent_to_grid_replicate_fill_matches_border():
    canvas = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    padded, (pad_top, pad_left) = pad_latent_to_grid(canvas, tile_l=8)
    # padded interior region equals original
    inner = padded[:, :, pad_top:pad_top + 4, pad_left:pad_left + 4]
    assert torch.equal(inner, canvas)
    # the column just left of the content equals the content's left border (replicate)
    left_border = canvas[:, :, :, 0]
    pad_col = padded[:, :, pad_top:pad_top + 4, pad_left - 1]
    assert torch.equal(pad_col, left_border)


def test_pad_latent_to_grid_zero_pad_when_aligned():
    canvas = torch.zeros(1, 4, 128, 256)  # both multiples of 128
    padded, (pad_top, pad_left) = pad_latent_to_grid(canvas, tile_l=128)
    assert padded.shape == canvas.shape
    assert (pad_top, pad_left) == (0, 0)
    assert padded is canvas  # no-op returns the same tensor
