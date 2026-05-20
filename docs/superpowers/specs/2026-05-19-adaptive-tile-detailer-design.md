# Adaptive Tiled Image Detailer — Design Spec

**Date:** 2026-05-19  
**Status:** Approved

## Problem

The existing `LLMTileSequentialDetailer` applies a single fixed `denoise` value to every tile. Flat regions (sky, clouds, walls) receive the same treatment as detailed regions (faces, fur, architecture). At moderate-to-high denoise values, the sampler injects hallucinated texture into flat areas — producing a "shadow image" or echo artifact. The fix is to lower denoise for flat tiles and reserve the full denoise budget for complex tiles.

## Goal

A new ComfyUI node that runs tile-by-tile denoising where the denoise strength is automatically derived from each tile's spatial complexity (latent variance). Flat tiles get low denoise (freeze), complex tiles get high denoise (sharpen). The resulting refinement has a pseudo-mosaic character: quiet areas are invisible, detailed areas pop.

## Node

**Class name:** `LLMAdaptiveTileDetailer`  
**Display name:** `Adaptive Tiled Image Detailer`  
**File:** `node_detailer_adaptive.py`  
**Category:** `image/generation`

## Inputs

| Parameter | Type | Default | Range / Notes |
|---|---|---|---|
| `model` | MODEL | — | |
| `upscaled_latent` | LATENT | — | |
| `positive` | CONDITIONING | — | |
| `negative` | CONDITIONING | — | |
| `seed` | INT | 0 | 0–0xffffffffffffffff, control_after_generate |
| `steps` | INT | 20 | 1–100 |
| `cfg` | FLOAT | 7.0 | 1.0–20.0, step 0.1 |
| `sampler_name` | SAMPLERS | — | |
| `scheduler` | SCHEDULERS | — | |
| `denoise_min` | FLOAT | 0.05 | 0.0–1.0, step 0.01 — flat tiles (sky, walls) |
| `denoise_max` | FLOAT | 0.35 | 0.0–1.0, step 0.01 — complex tiles (faces, fur) |
| `curve` | FLOAT | 1.5 | 0.1–5.0, step 0.1 — gamma exponent; >1 biases most tiles toward denoise_min |
| `tile_size` | INT | 1024 | 256–2048, step 8 |
| `overlap` | INT | 64 | 0–512, step 8 |

## Outputs

| Name | Type | Description |
|---|---|---|
| `refined_latent` | LATENT | Assembled canvas after adaptive tile sampling |
| `denoise_map` | IMAGE | Pixel-space heatmap; one solid-color rectangle per tile indicating its denoise value |

## Algorithm

### Pass 1 — Variance Scan (no GPU sampling)

For each tile position `(r, c)` in the center-anchored grid:

```python
tile_slice = canvas[:, :, y1:y2, x1:x2]
variance[r][c] = tile_slice.var(dim=[2, 3]).mean().item()
```

`var(dim=[2,3])` computes spatial variance per latent channel; `.mean()` averages across the 4 channels. Result is a scalar per tile — flat regions (uniform color/gradient) produce near-zero variance; regions with edges or texture produce higher values.

### Normalization + Curve

```python
v_min = min(all variances)
v_max = max(all variances)

if v_max == v_min:
    # Uniform image — cannot differentiate; default to full denoise
    t = 1.0
else:
    t = (variance - v_min) / (v_max - v_min)   # 0 = flattest, 1 = most complex

t_curved = t ** curve
denoise = denoise_min + t_curved * (denoise_max - denoise_min)
```

`curve > 1` applies a gamma that compresses the upper range — most tiles receive values closer to `denoise_min`, while only the highest-variance tiles approach `denoise_max`. `curve = 1.5` is the recommended default for a strong pseudo-mosaic effect.

### Pass 2 — Adaptive Sampling

Identical to `LLMTileSequentialDetailer`'s inner loop, with `denoise` replaced by the per-tile value from Pass 1. Each tile is logged:

```
[LLMAdaptiveTileDetailer] tile (r,c) var=0.0041 t=0.12 denoise=0.083
```

Blending uses the same `feather_blend_latent` from `utils`.

## Denoise Map Visualization

After Pass 1, construct a pixel-space IMAGE tensor `[1, H*8, W*8, 3]` filled black. For each tile, fill its pixel rectangle with a color interpolated by its normalized (pre-curved) `t` value:

- `t = 0.0` → blue `[0, 0, 1]`
- `t = 0.5` → green `[0, 1, 0]`
- `t = 1.0` → red `[1, 0, 0]`

Color is computed as linear HSV-style: `hue = (1 - t) * 240°`, saturation = 1, value = 1, converted to RGB. Tile boundaries are at latent coordinates × 8.

**Note:** The heatmap uses the pre-curve `t` (raw normalized variance), not `t_curved`. This shows where detail actually exists in the image, independent of the `curve` setting — making it the correct debug signal for tuning `denoise_min`/`denoise_max`.

This output is a mosaic of colored rectangles, visually revealing the algorithm's complexity map before sampling begins.

## Edge Cases

- **All tiles identical variance:** `t = 1.0` → all tiles get `denoise_max`
- **Overlap ≥ tile_size:** Clamped to `tile_l // 2` with a warning (same as existing node)
- **Single tile:** One tile covers the whole image; `t = 1.0` → `denoise_max`

## File Changes

| File | Action |
|---|---|
| `node_detailer_adaptive.py` | Create — contains `LLMAdaptiveTileDetailer` |
| `__init__.py` | Add node to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` |

No new utility functions. Reuses `feather_blend_latent` and `_compute_center_grid` from `utils`.
