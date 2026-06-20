# Edge Mode for Tile Detailers — Design Spec

**Date:** 2026-06-20
**Status:** Approved

## Problem

Both detailer nodes (`LLMTileSequentialDetailer`, `LLMAdaptiveTileDetailer`) build a
*centered* tile grid anchored to multiples of `tile_l`. The grid only covers a centered
region of `(cols+1)·tile_l × (rows+1)·tile_l`; the leftover margin around the edges (the
partial grid cells) is **never diffused**.

The current `crop_to_tiles` boolean offers only two outcomes:

- `True` — output is cropped down to the centered detailed region.
- `False` — output is full size, but the edge margins remain the **original undetailed**
  upscale.

There is no way to fully detail the whole image at its original size. The edge margins
stay un-refined.

## Goal

Let the user choose how the tile-grid edges are handled, including a new mode that details
the entire image at its original size. Replace the `crop_to_tiles` boolean with a 3-way
`edge_mode` selector.

These nodes operate in **latent space** (1/8 resolution); there is no VAE here. Padding
fills are therefore done by edge-replication of the latent tensor — the latent-space
equivalent of "stretch the border pixels outward."

## Interface change (both nodes)

Replace the `crop_to_tiles` BOOLEAN input with:

```python
"edge_mode": (["center", "crop", "pad"], {"default": "center"})
```

| Mode | What runs | Output size | Edge margins |
|---|---|---|---|
| `center` | diffuse centered grid only | original | left as original upscale (undetailed) |
| `crop` | diffuse centered grid only | cropped to detailed region | n/a (cropped away) |
| `pad` | pad to full grid, diffuse everything, crop back | original | fully detailed |

- `center` = today's `crop_to_tiles=False` (current default; default preserved).
- `crop` = today's `crop_to_tiles=True`.
- `pad` = new full-coverage behavior.

**Compatibility:** changing the boolean to a combo input means existing saved workflows
reset that widget to the default. Defaulting to `center` preserves the current default
output (full size, undetailed margins), so default behavior is unchanged.

**Why a mode, not a threshold:** covering any leftover margin with fixed `tile_l`-wide
tiles costs one whole extra tile-row of compute (full coverage needs `ceil(W/tile_l)`
tiles; the existing tile count can only span `(W//tile_l)·tile_l` gap-free). Whether that
is worth it depends on how big the margin is — a decision left to the user via the mode,
rather than an auto-guessed heuristic. If the leftover sliver is small and not worth a
full tile, pick `center`; if it should be covered, pick `pad`.

## New shared helper — `utils/image_utils.py`

```
pad_latent_to_grid(canvas, tile_l) -> (padded_canvas, (pad_top, pad_left))
```

- Target dims: `Wp = ceil(W/tile_l)·tile_l`, `Hp = ceil(H/tile_l)·tile_l`.
- Pad split symmetrically per axis: `pad_left = (Wp-W)//2`, `pad_right = Wp-W-pad_left`
  (same for vertical).
- Fill via `torch.nn.functional.pad(canvas, (l, r, t, b), mode="replicate")` — repeats the
  border latent row/column outward.
- Per-axis independence: a dimension already a multiple of `tile_l` gets zero pad, so an
  axis that does not need padding is never padded.
- Returns the padded canvas and the `(pad_top, pad_left)` offsets needed to crop back.
- Exported from `utils/__init__.py`.

Because the padded dims are exact multiples of `tile_l`, the centered grid's
`start_x`/`start_y` resolve to 0 and the tiles cover the **entire** padded canvas — so the
original region is guaranteed fully covered regardless of the overlap setting.

## Node flow (both `detail` methods)

1. Read `_, _, H, W`; remember original `H0, W0`.
2. If `edge_mode == "pad"`:
   `canvas, (pad_top, pad_left) = pad_latent_to_grid(canvas, tile_l)`, then recompute
   `H, W` from the padded canvas. Otherwise `pad_top = pad_left = 0`.
3. Grid build (`_compute_center_grid`, `_compute_tile_coords`), sampling loop, and
   `feather_blend_latent`: **unchanged**, run on `canvas`.
4. Final output selection:
   - `crop` → existing centered-grid crop (unchanged):
     `canvas[:, :, y1_c:y2_c, x1_c:x2_c]`.
   - `pad` → crop back to the original region:
     `canvas[:, :, pad_top:pad_top+H0, pad_left:pad_left+W0]`.
   - `center` → full canvas as-is (unchanged).
5. **Adaptive node only:** apply the same `pad`/`crop` slicing (×8 for pixel-resolution
   maps) to `denoise_map_img` and `scoring_map_img` so they match the returned canvas.

## Cost

Worst case is +1 tile column and +1 row (only in `pad` mode). For a large multi-tile
upscale that is roughly ≤2x; for a tiny grid (e.g. 1×1 → 2×2) it can be up to ~4x.
Acceptable since it only fires when the user explicitly selects `pad`.

## Testing

- `pad_latent_to_grid`:
  - pads to correct `tile_l` multiples;
  - symmetric pad split;
  - replicate fill matches the border latent values;
  - zero-pad (and zero offsets) when both dims already aligned;
  - returns correct `(pad_top, pad_left)` offsets.
- Sequential node `pad`: output shape == **original** input shape; former-margin regions
  differ from the input (i.e. they were diffused).
- Sequential node `crop` and `center`: outputs unchanged vs current `True`/`False`
  behavior (regression guards).
- Adaptive node `pad`: canvas and both map outputs match the original input dimensions.

## Docs

Update the README section that documents `crop_to_tiles` to describe `edge_mode` and its
three values.
