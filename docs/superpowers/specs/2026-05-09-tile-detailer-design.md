# Tile Detailer — Design Spec

**Date:** 2026-05-09  
**Branch:** feat/tile-detailer  
**Node class:** `LLMTileSequentialDetailer`  
**Display name:** `LLM Tile Sequential Detailer`

---

## Problem

`[BETA] Tiled Diffusion` causes severe VRAM saturation (~40 s/it) on Windows with shared VRAM because it patches the entire 4K model simultaneously. The fix is a custom sequential sampler that processes one tile at a time, flushing GPU buffers between tiles via `soft_empty_cache()`.

---

## Goal

A ComfyUI node that takes an upscaled latent and runs partial denoising (0.20–0.35) tile-by-tile, injecting detail (texture, sharpness) without changing layout or colors. Expected throughput: ~1.2 it/s, ~8 GB VRAM, ~25 s for a full 4K pass.

---

## Workflow Position

```
[Upscaled Image] → VAE Encode (Tiled) → upscaled_latent
                                              ↓
[CLIPTextEncode] → positive    LLMTileSequentialDetailer
[CLIPTextEncode] → negative         ↓
                              refined_latent
                                    ↓
                         VAE Decode (Tiled) → Final Image
```

VAE Encode (Tiled) and VAE Decode (Tiled) are standard ComfyUI core nodes. The detailer node has no VAE dependency.

---

## Files Changed

| File | Change |
|---|---|
| `node_detailer.py` | New — contains `LLMTileSequentialDetailer` |
| `utils/image_utils.py` | Add `feather_blend_latent()` |
| `utils/__init__.py` | Export `feather_blend_latent` |
| `__init__.py` | Import and merge new node mappings |
| `tests/test_detailer.py` | New — unit tests for blend function and grid math |

---

## Node Definition

**Category:** `sampling/detailers`  
**Function:** `detail`  
**Return:** `("LATENT",)` named `("refined_latent",)`

### INPUT_TYPES

| Name | Type | Default | Range |
|---|---|---|---|
| `model` | `MODEL` | — | — |
| `upscaled_latent` | `LATENT` | — | — |
| `positive` | `CONDITIONING` | — | — |
| `negative` | `CONDITIONING` | — | — |
| `seed` | `INT` | 0 | 0 – 0xffffffffffffffff |
| `steps` | `INT` | 20 | 1 – 100 |
| `cfg` | `FLOAT` | 7.0 | 1.0 – 20.0, step 0.1 |
| `sampler_name` | `SAMPLER` dropdown | — | `KSampler.SAMPLERS` |
| `scheduler` | `SCHEDULER` dropdown | — | `KSampler.SCHEDULERS` |
| `denoise` | `FLOAT` | 0.25 | 0.05 – 1.0, step 0.01 |
| `tile_size` | `INT` | 1024 | 256 – 2048, step 8 |
| `overlap` | `INT` | 64 | 0 – 512, step 8 |

---

## `detail` Method Logic

### Latent-space units

All grid math uses latent-space coordinates:

```python
tile_l    = tile_size // 8
overlap_l = min(overlap // 8, tile_l // 2)   # guarded against zero-stride
stride    = tile_l - overlap_l
```

If `overlap // 8 >= tile_l`, emit a warning and clamp to `tile_l // 2`.

### Center-anchored grid

```python
cols = max(1, (W - overlap_l) // stride)
rows = max(1, (H - overlap_l) // stride)

start_x = (W - (cols * stride + overlap_l)) // 2
start_y = (H - (rows * stride + overlap_l)) // 2
```

`start_x` / `start_y` may be slightly negative when the image doesn't divide evenly; all coordinates are clamped before use.

### Sequential tile loop

Iterates `rows + 1` × `cols + 1` to guarantee far-edge coverage:

```python
canvas = upscaled_latent["samples"].clone()   # never mutate the input

for r in range(rows + 1):
    for c in range(cols + 1):
        y1 = max(0, start_y + r * stride)
        x1 = max(0, start_x + c * stride)
        y2 = min(H, y1 + tile_l)
        x2 = min(W, x1 + tile_l)

        if y2 <= y1 or x2 <= x1:
            continue                              # zero-area tile, skip

        tile_latent = canvas[:, :, y1:y2, x1:x2].clone()
        tile_seed   = seed + r * (cols + 1) + c

        refined = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative,
            {"samples": tile_latent},
            denoise=denoise
        )["samples"]

        feather_blend_latent(canvas, refined, y1, x1, overlap_l,
                             has_left=(x1 > 0), has_top=(y1 > 0))

        comfy.model_management.soft_empty_cache()
        pbar.update(1)

return ({"samples": canvas},)
```

Seed is row-major per tile (`seed + tile_index`), matching existing node convention.

### Noise for `comfy.sample.sample`

`comfy.sample.prepare_noise(tile_latent, tile_seed, None)` — same pattern as `node.py`.

---

## `feather_blend_latent`

Lives in `utils/image_utils.py`. Signature:

```python
def feather_blend_latent(canvas, refined, y1, x1, overlap_l, has_left, has_top):
```

- `canvas`: `[B, C, H, W]` full latent being built up
- `refined`: `[B, C, tile_h, tile_w]` output from sampler (may be smaller than `tile_l` at edges)
- `y1, x1`: top-left insertion point in canvas coordinates
- `overlap_l`: overlap in latent pixels

**Write order:**

1. Hard-write the full `refined` region into canvas (interior + overlap)
2. If `has_left` and `overlap_l > 0`: apply linear ramp `[0→1]` across `overlap_l` columns, blending from canvas value toward refined value
3. If `has_top` and `overlap_l > 0`: apply linear ramp `[0→1]` across `overlap_l` rows
4. If both: corner uses `min(alpha_x, alpha_y)` — same strategy as `blend_and_place_tile`

Ramps are `torch.linspace(0.0, 1.0, overlap_l)` on the canvas device.

---

## Edge Cases

| Situation | Behaviour |
|---|---|
| Latent smaller than `tile_l` | `cols=1, rows=1`, single tile covers whole latent — normal img2img |
| `overlap >= tile_size // 2` | Clamped to `tile_l // 2` with warning printed |
| Zero-area tile after clamping | Skipped with `continue` |
| Single-tile image | Works identically to a standard KSampler img2img pass |

---

## Testing

**`tests/test_detailer.py`** — no ComfyUI runtime required.

### `test_feather_blend_latent_left_edge`
- Two synthetic `[1, 4, 8, 16]` latents
- Call `feather_blend_latent` with `has_left=True, has_top=False`
- Assert overlap columns ramp smoothly (not a hard cut); assert interior is fully from `refined`

### `test_feather_blend_latent_top_edge`
- Same structure, `has_left=False, has_top=True`

### `test_center_anchored_grid`
- Input: 1000×2048 portrait → latent `W=125, H=256`, `tile_size=1024, overlap=64`
- Assert `cols`, `rows`, `start_x`, `start_y` match expected values
- Assert all clamped tile coordinates are within `[0, W]` / `[0, H]`
