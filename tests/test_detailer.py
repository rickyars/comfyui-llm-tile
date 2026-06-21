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
        edge_mode="center",
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
            edge_mode="center",
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
    assert "edge_mode" in required
    assert "crop_to_tiles" not in required


# ---------------------------------------------------------------------------
# Task: pad_latent_to_grid
# ---------------------------------------------------------------------------

from utils.image_utils import (
    pad_latent_to_grid, _compute_center_grid, _compute_tile_coords,
)


def _tile_anchors(coords, tile_l):
    # The anchor (top-left of the un-grown tile) is exact regardless of overlap.
    return sorted({(y2 - tile_l, x2 - tile_l) for (y1, x1, y2, x2) in coords})


def test_pad_latent_to_grid_pads_to_tile_multiple():
    canvas = torch.zeros(1, 4, 100, 140)  # neither dim a multiple of 64
    padded, (pad_top, pad_left) = pad_latent_to_grid(canvas, tile_l=64)
    # Padded dims are exact tile multiples so the grid tiles the whole canvas.
    assert padded.shape[2] % 64 == 0
    assert padded.shape[3] % 64 == 0
    # H=100: 1 center tile (64), margins 18 each -> pad 46 each -> 192
    # W=140: 2 center tiles (128), margins 6 each -> pad 58 each -> 256
    assert padded.shape[2] == 192
    assert padded.shape[3] == 256


def test_pad_latent_to_grid_preserves_center_grid():
    # The core fix: pad mode must keep the center-mode tile positions and only
    # add edge tiles, so the detail grid looks the same as center mode.
    W0, H0, tile_l, overlap_l = 1008, 1024, 128, 8
    canvas = torch.zeros(1, 4, H0, W0)

    c, r = _compute_center_grid(W0, H0, tile_l, overlap_l)
    center_anchors = _tile_anchors(
        _compute_tile_coords(W0, H0, tile_l, c, r, overlap_l), tile_l)

    padded, (pad_top, pad_left) = pad_latent_to_grid(canvas, tile_l)
    _, _, Hp, Wp = padded.shape
    c2, r2 = _compute_center_grid(Wp, Hp, tile_l, overlap_l)
    padded_anchors = {
        (ay - pad_top, ax - pad_left)
        for (ay, ax) in _tile_anchors(
            _compute_tile_coords(Wp, Hp, tile_l, c2, r2, overlap_l), tile_l)
    }

    # Every center-mode tile anchor is still present at the same original coords.
    assert set(center_anchors).issubset(padded_anchors)
    # Padded canvas is fully tileable.
    assert Wp % tile_l == 0 and Hp % tile_l == 0


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


# ---------------------------------------------------------------------------
# Task: edge_mode (sequential)
# ---------------------------------------------------------------------------

def test_edge_mode_pad_returns_original_shape():
    from node_detailer import LLMTileSequentialDetailer
    node = LLMTileSequentialDetailer()
    # 40x44 latent, tile_l=16 (tile_size=128) -> not tile-aligned
    latent = torch.randn(1, 4, 40, 44)
    out = node.detail(
        model=_make_model_mock(),
        upscaled_latent={"samples": latent.clone()},
        positive=[], negative=[],
        seed=0, steps=20, cfg=7.0,
        sampler_name="euler", scheduler="normal",
        denoise=0.25, tile_size=128, overlap=0,
        edge_mode="pad",
        noise_type="gaussian", eta=0.0,
    )
    assert out[0]["samples"].shape == latent.shape


def test_edge_mode_pad_details_former_margins():
    # sample_custom (stubbed) returns latent.clone(), so diffused regions are
    # unchanged vs input. To prove the margin tiles RAN, count sample_custom
    # calls: pad mode must invoke more tiles than center mode for the same input.
    from node_detailer import LLMTileSequentialDetailer
    node = LLMTileSequentialDetailer()
    latent = torch.randn(1, 4, 40, 44)
    common = dict(
        model=_make_model_mock(), positive=[], negative=[],
        seed=0, steps=20, cfg=7.0, sampler_name="euler", scheduler="normal",
        denoise=0.25, tile_size=128, overlap=0, noise_type="gaussian", eta=0.0,
    )

    comfy.sample.sample_custom.reset_mock()
    node.detail(upscaled_latent={"samples": latent.clone()}, edge_mode="center", **common)
    center_calls = comfy.sample.sample_custom.call_count

    comfy.sample.sample_custom.reset_mock()
    node.detail(upscaled_latent={"samples": latent.clone()}, edge_mode="pad", **common)
    pad_calls = comfy.sample.sample_custom.call_count

    assert pad_calls > center_calls


def test_edge_mode_crop_shrinks_output():
    from node_detailer import LLMTileSequentialDetailer
    node = LLMTileSequentialDetailer()
    latent = torch.randn(1, 4, 40, 44)
    out = node.detail(
        model=_make_model_mock(),
        upscaled_latent={"samples": latent.clone()},
        positive=[], negative=[],
        seed=0, steps=20, cfg=7.0,
        sampler_name="euler", scheduler="normal",
        denoise=0.25, tile_size=128, overlap=0,
        edge_mode="crop",
        noise_type="gaussian", eta=0.0,
    )
    s = out[0]["samples"].shape
    # 40x44 latent, tile_l=16 -> centered 2x2 grid crops to exactly 32x32
    assert s[2] == 32 and s[3] == 32


def test_edge_mode_center_keeps_original_shape():
    from node_detailer import LLMTileSequentialDetailer
    node = LLMTileSequentialDetailer()
    latent = torch.randn(1, 4, 40, 44)
    out = node.detail(
        model=_make_model_mock(),
        upscaled_latent={"samples": latent.clone()},
        positive=[], negative=[],
        seed=0, steps=20, cfg=7.0,
        sampler_name="euler", scheduler="normal",
        denoise=0.25, tile_size=128, overlap=0,
        edge_mode="center",
        noise_type="gaussian", eta=0.0,
    )
    assert out[0]["samples"].shape == latent.shape


def test_edge_mode_pad_blends_diffused_content_into_margins():
    # Round-trip check: prove pad mode actually writes diffused tile content
    # into the (cropped-back) output, not just that it returns the right shape.
    # Override the stub so each tile comes back clearly modified (+100), then
    # confirm the original-size output differs from the input everywhere.
    from node_detailer import LLMTileSequentialDetailer
    node = LLMTileSequentialDetailer()
    latent = torch.randn(1, 4, 40, 44)

    original = comfy.sample.sample_custom
    comfy.sample.sample_custom = MagicMock(
        side_effect=lambda model, noise, cfg, sampler, sigmas, pos, neg, lat, **kw: lat + 100.0
    )
    try:
        out = node.detail(
            model=_make_model_mock(),
            upscaled_latent={"samples": latent.clone()},
            positive=[], negative=[],
            seed=0, steps=20, cfg=7.0,
            sampler_name="euler", scheduler="normal",
            denoise=0.25, tile_size=128, overlap=0,
            edge_mode="pad",
            noise_type="gaussian", eta=0.0,
        )
    finally:
        comfy.sample.sample_custom = original

    result = out[0]["samples"]
    assert result.shape == latent.shape
    # Every pixel of the original region is covered by a diffused tile, so the
    # whole output should be shifted by +100 vs the input (no undetailed margin).
    assert torch.all(result > latent + 50.0)
