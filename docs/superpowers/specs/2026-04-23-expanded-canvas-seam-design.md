# Expanded Canvas + ControlNet Seam Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix tile seams by expanding the generation canvas per-tile so ControlNet can inpaint the overlap zone, then extracting only the new-content zone and crossfading into the already-placed neighbor.

**Architecture:** Each tile generates at `(tile_width + overlap_x) × (tile_height + overlap_y)` when neighbors exist. ControlNet constrains the overlap zone to match the neighbor. The non-overlap zone is extracted and placed in the canvas. The discarded overlap zone (ControlNet-matched) is used to crossfade over the neighbor's already-placed edge — hiding any residual mismatch. Without ControlNet the tile generates at exactly `tile_width × tile_height` and is hard-placed (test mode).

**Tech Stack:** PyTorch, ComfyUI VAE/ControlNet/KSampler APIs, existing `utils/image_utils.py`

**Files:** `node.py`, `node_advanced.py`, `utils/image_utils.py`, `utils/__init__.py`

---

## Section 1: Generation Canvas Geometry

For each tile at grid position `(x, y)`:

```python
has_left = x > 0  # or seamlessX last-column wrapping
has_top  = y > 0  # or seamlessY last-row wrapping

# With ControlNet
gen_width  = tile_width  + (overlap_x if has_left else 0)
gen_height = tile_height + (overlap_y if has_top  else 0)

# Without ControlNet (test mode)
gen_width  = tile_width
gen_height = tile_height
```

`working_tensor` shape: `(1, gen_height, gen_width, 3)`, filled with zeros then neighbor pixels:

| Zone | Source | Target in working_tensor |
|------|--------|--------------------------|
| Left strip | `canvas[pos_y:pos_y+tile_h, pos_x-overlap_x:pos_x]` | `[:, :, 0:overlap_x, :]` (shifted down by overlap_y if has_top) |
| Top strip | `canvas[pos_y-overlap_y:pos_y, pos_x:pos_x+tile_w]` | `[:, 0:overlap_y, :, :]` (shifted right by overlap_x if has_left) |
| Corner | `canvas[pos_y-overlap_y:pos_y, pos_x-overlap_x:pos_x]` | `[:, 0:overlap_y, 0:overlap_x, :]` |
| SeamlessX wrap | first column's left edge | right strip of working_tensor |
| SeamlessY wrap | first row's top edge | bottom strip of working_tensor |

For seamless wrapping edges, `gen_width` / `gen_height` grow by an additional `overlap_x` / `overlap_y` on the wrap side.

---

## Section 2: Extraction

After `vae.decode(samples)`:

```python
start_x = overlap_x if has_left else 0
start_y = overlap_y if has_top  else 0

extracted = decoded[0,
    start_y : start_y + tile_height,
    start_x : start_x + tile_width, :]
```

`extracted` is always exactly `tile_height × tile_width`. The canvas placement position is unchanged: `canvas[pos_y : pos_y+tile_height, pos_x : pos_x+tile_width]`.

---

## Section 3: Blend

The ControlNet-matched overlap zone (the pixels we just extracted _past_) is used to crossfade over the neighbor's already-placed edge before hard-placing `extracted`.

**Left seam** (has_left, with ControlNet):
```python
# matched_left = generated_tile[start_y:start_y+tile_h, 0:overlap_x]  — ControlNet-matched
alpha = torch.linspace(0.0, 1.0, overlap_x)  # shape [overlap_x]
canvas[pos_y:pos_y+tile_h, pos_x-overlap_x:pos_x] = (
    (1 - alpha) * canvas[pos_y:pos_y+tile_h, pos_x-overlap_x:pos_x]
  +      alpha  * matched_left
)
```

**Top seam** (has_top, with ControlNet):
```python
# matched_top = generated_tile[0:overlap_y, start_x:start_x+tile_w]
alpha = torch.linspace(0.0, 1.0, overlap_y).view(-1, 1)
canvas[pos_y-overlap_y:pos_y, pos_x:pos_x+tile_w] = (
    (1 - alpha) * canvas[pos_y-overlap_y:pos_y, pos_x:pos_x+tile_w]
  +      alpha  * matched_top
)
```

**Corner** (has_left AND has_top, with ControlNet):
```python
# matched_corner = generated_tile[0:overlap_y, 0:overlap_x]
alpha_x = torch.linspace(0.0, 1.0, overlap_x).view(1, -1)
alpha_y = torch.linspace(0.0, 1.0, overlap_y).view(-1, 1)
alpha   = torch.min(alpha_x.expand(overlap_y, -1), alpha_y.expand(-1, overlap_x))
canvas[pos_y-overlap_y:pos_y, pos_x-overlap_x:pos_x] = (
    (1 - alpha) * canvas[pos_y-overlap_y:pos_y, pos_x-overlap_x:pos_x]
  +      alpha  * matched_corner
)
```

Then hard-place:
```python
canvas[pos_y:pos_y+tile_height, pos_x:pos_x+tile_width] = extracted
```

**Without ControlNet (test mode):** skip all blend steps, just hard-place `extracted` (= full tile at tile_width × tile_height).

---

## Section 4: ControlNet-Optional Behavior

| | With ControlNet | Without ControlNet (test mode) |
|---|---|---|
| `gen_width` | `tile_width + overlap_x` (if has_left) | `tile_width` always |
| `gen_height` | `tile_height + overlap_y` (if has_top) | `tile_height` always |
| `working_tensor` | filled with neighbor pixels | skip construction entirely |
| `apply_controlnet_to_conditioning` | called with working_tensor | skipped (controlnet=None) |
| latent shape | `vae.encode(working_tensor).shape` | compute directly: `(1, 4, gen_height//8, gen_width//8)` |
| Extraction | `tile[start_y:, start_x:]` | `tile[0:tile_h, 0:tile_w]` |
| Blend | crossfade over neighbor edge | hard-place only |
| Seam quality | ControlNet + blend | pixel-level hard cut |

`controlnet=None` is already in the `optional` INPUT_TYPES section — no UI change needed.

---

## Section 5: Seamless Wrapping

Unchanged from current logic:

- **SeamlessX last column**: `working_tensor` right strip = `canvas[pos_y:, 0:overlap_x]` (first column). `gen_width` grows by additional `overlap_x`. Extract from middle: `tile[:, overlap_x : overlap_x+tile_width]`. After all tiles placed, trim: `final_tensor = final_tensor[:, :, :final_width-overlap_x, :]`.
- **SeamlessY last row**: same vertically.

---

## Key Invariants

- Final canvas is always `grid_width * tile_width × grid_height * tile_height` before seamless trim.
- `extracted` is always exactly `tile_height × tile_width` regardless of canvas expansion.
- Blend writes into already-placed neighbor zone (never into the new tile's zone) — no echo artifact.
- `blend_tile_into_canvas` in `image_utils.py` is replaced by the inline blend + hard-place described above (or refactored into a new `blend_and_place_tile` function with signature matching the new logic).
