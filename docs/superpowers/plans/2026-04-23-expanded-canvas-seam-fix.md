# Expanded Canvas Seam Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix tile seams by expanding the generation canvas per-tile so ControlNet constrains the overlap zone from neighbor content, then extracting only the new-content pixels and crossfading into the neighbor's already-placed edge.

**Architecture:** Each tile with neighbors generates at `(tile_width + overlap_x) × (tile_height + overlap_y)` when ControlNet is active. ControlNet constrains the left/top overlap strips to match the neighbor. Only the non-overlap zone is extracted and hard-placed. The discarded ControlNet-matched strips are used to crossfade over the neighbor's already-placed edge. Without ControlNet, tiles generate at `tile_width × tile_height` and are hard-placed (test mode).

**Tech Stack:** PyTorch, ComfyUI VAE/ControlNet/KSampler APIs, existing `utils/image_utils.py`

---

## File Map

| File | Change |
|------|--------|
| `utils/image_utils.py` | Replace `blend_tile_into_canvas` with `blend_and_place_tile` |
| `utils/__init__.py` | Update export |
| `node.py` | Per-tile gen size, remove VAE shape probe, correct extraction, new blend call |
| `node_advanced.py` | Same as `node.py` |
| `tests/test_blend_and_place.py` | New test file |

---

### Task 1: `blend_and_place_tile` in `utils/image_utils.py`

**Files:**
- Modify: `utils/image_utils.py`
- Create: `tests/test_blend_and_place.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_blend_and_place.py`:

```python
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
    assert canvas[0, :, 870:1024, :].mean().item() == pytest.approx(1.0, abs=1e-4)
    assert canvas[0, :, 1024:2048, :].mean().item() == pytest.approx(0.5, abs=1e-4)


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
```

- [ ] **Step 2: Run tests, verify they fail**

```
"D:/miniconda3/envs/comfyui/python.exe" -m pytest tests/test_blend_and_place.py -v --import-mode=importlib
```

Expected: FAIL — `ImportError: cannot import name 'blend_and_place_tile'`

- [ ] **Step 3: Replace `blend_tile_into_canvas` with `blend_and_place_tile` in `utils/image_utils.py`**

Delete the entire `blend_tile_into_canvas` function (lines 13–54). Add in its place:

```python
def blend_and_place_tile(canvas, generated_tile, pos_x, pos_y,
                          tile_width, tile_height, overlap_x, overlap_y,
                          has_left, has_top, controlnet_active):
    """
    Extract the new-content zone from generated_tile, crossfade the ControlNet-matched
    overlap strips into the neighbor's already-placed edge, then hard-place the extracted zone.

    generated_tile shape: [gen_h, gen_w, 3] where gen_h = tile_height + (overlap_y if has_top else 0)
    and gen_w = tile_width + (overlap_x if has_left else 0), when controlnet_active is True.
    Without ControlNet, gen_h = tile_height, gen_w = tile_width (no expansion).
    """
    tile_cpu = generated_tile.cpu() if generated_tile.is_cuda else generated_tile

    start_x = (overlap_x if has_left else 0) if controlnet_active else 0
    start_y = (overlap_y if has_top else 0) if controlnet_active else 0
    extracted = tile_cpu[start_y:start_y + tile_height, start_x:start_x + tile_width, :]

    if controlnet_active:
        if has_left and overlap_x > 0:
            matched_left = tile_cpu[start_y:start_y + tile_height, 0:overlap_x, :]
            alpha = torch.linspace(0.0, 1.0, overlap_x).view(1, overlap_x, 1)
            zone = canvas[0, pos_y:pos_y + tile_height, pos_x - overlap_x:pos_x, :].clone()
            canvas[0, pos_y:pos_y + tile_height, pos_x - overlap_x:pos_x, :] = (
                (1.0 - alpha) * zone + alpha * matched_left
            )

        if has_top and overlap_y > 0:
            matched_top = tile_cpu[0:overlap_y, start_x:start_x + tile_width, :]
            alpha = torch.linspace(0.0, 1.0, overlap_y).view(overlap_y, 1, 1)
            zone = canvas[0, pos_y - overlap_y:pos_y, pos_x:pos_x + tile_width, :].clone()
            canvas[0, pos_y - overlap_y:pos_y, pos_x:pos_x + tile_width, :] = (
                (1.0 - alpha) * zone + alpha * matched_top
            )

        if has_left and has_top and overlap_x > 0 and overlap_y > 0:
            matched_corner = tile_cpu[0:overlap_y, 0:overlap_x, :]
            alpha_x = torch.linspace(0.0, 1.0, overlap_x).view(1, overlap_x, 1).expand(overlap_y, -1, 1)
            alpha_y = torch.linspace(0.0, 1.0, overlap_y).view(overlap_y, 1, 1).expand(-1, overlap_x, 1)
            alpha = torch.min(alpha_x, alpha_y)
            zone = canvas[0, pos_y - overlap_y:pos_y, pos_x - overlap_x:pos_x, :].clone()
            canvas[0, pos_y - overlap_y:pos_y, pos_x - overlap_x:pos_x, :] = (
                (1.0 - alpha) * zone + alpha * matched_corner
            )

    canvas[0, pos_y:pos_y + tile_height, pos_x:pos_x + tile_width, :] = extracted
```

- [ ] **Step 4: Run tests, verify they pass**

```
"D:/miniconda3/envs/comfyui/python.exe" -m pytest tests/test_blend_and_place.py -v --import-mode=importlib
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```
git add utils/image_utils.py tests/test_blend_and_place.py
git commit -m "feat: replace blend_tile_into_canvas with blend_and_place_tile"
```

---

### Task 2: Update `utils/__init__.py`

**Files:**
- Modify: `utils/__init__.py`

- [ ] **Step 1: Replace the export**

Change line 2 of `utils/__init__.py`:

```python
# Before
from .image_utils import gaussian_blend_tiles, resize_mask_to_latent, blend_tile_into_canvas

# After
from .image_utils import gaussian_blend_tiles, resize_mask_to_latent, blend_and_place_tile
```

- [ ] **Step 2: Verify**

```
"D:/miniconda3/envs/comfyui/python.exe" -c "from utils import blend_and_place_tile; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```
git add utils/__init__.py
git commit -m "chore: update utils export for blend_and_place_tile"
```

---

### Task 3: Update `node.py`

**Files:**
- Modify: `node.py`

**Background for the implementer:** `node.py` has a tile generation loop. The current loop has two bugs: (1) `gen_width`/`gen_height` are fixed before the loop and don't account for neighbors, (2) `vae.encode(working_tensor)` is called just to probe the latent shape — expensive and unnecessary. The fixes: compute gen size per-tile inside the loop based on `has_left_neighbor`, `has_top_neighbor`, and `controlnet`, compute latent shape mathematically, and replace `blend_tile_into_canvas` with `blend_and_place_tile`.

- [ ] **Step 1: Update the import at top of `node.py`**

```python
from .utils import parse_tile_prompts
from .utils import apply_controlnet_to_conditioning, blend_and_place_tile
```

- [ ] **Step 2: Delete the pre-loop gen_width/gen_height lines**

Remove these two lines (currently after the print statements, before `final_tensor = ...`):

```python
# Delete these:
gen_width = ((tile_width + 7) // 8) * 8
gen_height = ((tile_height + 7) // 8) * 8
```

- [ ] **Step 3: Replace the full `for y / for x` loop body**

Replace everything from `for y in range(grid_height):` through the end of the loop (including pbar.update) with:

```python
        for y in range(grid_height):
            for x in range(grid_width):
                idx = y * grid_width + x
                current_prompt = tile_prompts[idx]
                current_seed = seed + idx
                final_pos_x = x * tile_width
                final_pos_y = y * tile_height

                has_left_neighbor = x > 0
                has_top_neighbor = y > 0
                controlnet_active = controlnet is not None

                # Expand generation canvas only when ControlNet can use the overlap zone
                if controlnet_active:
                    gen_w = (tile_width
                             + (overlap_x if has_left_neighbor else 0)
                             + (overlap_x if seamlessX and x == grid_width - 1 else 0))
                    gen_h = (tile_height
                             + (overlap_y if has_top_neighbor else 0)
                             + (overlap_y if seamlessY and y == grid_height - 1 else 0))
                else:
                    gen_w, gen_h = tile_width, tile_height

                # Align to multiple of 8 for VAE latent compatibility
                gen_w8 = ((gen_w + 7) // 8) * 8
                gen_h8 = ((gen_h + 7) // 8) * 8

                print(f"Tile ({x+1},{y+1}) gen canvas: {gen_w8}x{gen_h8}, canvas pos: ({final_pos_x},{final_pos_y})")
                print(f"Generating tile ({x+1},{y+1}) prompt: {current_prompt}")

                combined_prompt = f"{current_prompt} {global_positive}" if global_positive else current_prompt
                pos_cond = clip.encode_from_tokens_scheduled(clip.tokenize(combined_prompt))
                neg_cond = clip.encode_from_tokens_scheduled(clip.tokenize(global_negative))

                if controlnet_active:
                    working_tensor = torch.zeros((1, gen_h8, gen_w8, 3), dtype=torch.float32)

                    if has_left_neighbor:
                        source_x = final_pos_x - overlap_x
                        source_start_y = final_pos_y
                        source_end_y = min(final_pos_y + tile_height, final_height)
                        source_height = source_end_y - source_start_y
                        target_start_y = overlap_y if has_top_neighbor else 0
                        copy_height = min(source_height, gen_h8 - target_start_y)
                        if copy_height > 0:
                            working_tensor[0,
                                target_start_y:target_start_y + copy_height, :overlap_x, :
                            ] = final_tensor[0,
                                source_start_y:source_start_y + copy_height,
                                source_x:source_x + overlap_x, :]

                    if seamlessX and x == grid_width - 1 and overlap_x > 0:
                        wrap_target_x = overlap_x + tile_width
                        source_start_y = final_pos_y
                        source_end_y = min(final_pos_y + tile_height, final_height)
                        source_height = source_end_y - source_start_y
                        target_start_y = overlap_y if has_top_neighbor else 0
                        copy_height = min(source_height, gen_h8 - target_start_y)
                        if copy_height > 0 and wrap_target_x + overlap_x <= gen_w8:
                            working_tensor[0,
                                target_start_y:target_start_y + copy_height,
                                wrap_target_x:wrap_target_x + overlap_x, :
                            ] = final_tensor[0,
                                source_start_y:source_start_y + copy_height, 0:overlap_x, :]

                    if seamlessY and y == grid_height - 1 and overlap_y > 0:
                        wrap_target_y = overlap_y + tile_height
                        source_start_x = final_pos_x
                        source_end_x = min(final_pos_x + tile_width, final_width)
                        source_width = source_end_x - source_start_x
                        target_start_x = overlap_x if has_left_neighbor else 0
                        copy_width = min(source_width, gen_w8 - target_start_x)
                        if copy_width > 0 and wrap_target_y + overlap_y <= gen_h8:
                            working_tensor[0,
                                wrap_target_y:wrap_target_y + overlap_y,
                                target_start_x:target_start_x + copy_width, :
                            ] = final_tensor[0,
                                0:overlap_y, source_start_x:source_start_x + copy_width, :]

                    if has_top_neighbor:
                        source_y = final_pos_y - overlap_y
                        source_start_x = final_pos_x
                        source_end_x = min(final_pos_x + tile_width, final_width)
                        source_width = source_end_x - source_start_x
                        target_start_x = overlap_x if has_left_neighbor else 0
                        copy_width = min(source_width, gen_w8 - target_start_x)
                        if copy_width > 0:
                            working_tensor[0,
                                :overlap_y, target_start_x:target_start_x + copy_width, :
                            ] = final_tensor[0,
                                source_y:source_y + overlap_y,
                                source_start_x:source_start_x + copy_width, :]

                    if has_left_neighbor and has_top_neighbor:
                        corner_source_x = final_pos_x - overlap_x
                        corner_source_y = final_pos_y - overlap_y
                        working_tensor[0, :overlap_y, :overlap_x, :] = final_tensor[0,
                            corner_source_y:corner_source_y + overlap_y,
                            corner_source_x:corner_source_x + overlap_x, :]

                    conditioning = apply_controlnet_to_conditioning(
                        positive=pos_cond, negative=neg_cond,
                        control_net=controlnet, image=working_tensor,
                        strength=controlnet_strength, start_percent=0.0, end_percent=1.0, vae=vae
                    )
                else:
                    conditioning = (pos_cond, neg_cond)

                # Compute latent shape mathematically — no vae.encode() shape probe
                latent_image = torch.zeros((1, 4, gen_h8 // 8, gen_w8 // 8), device=device)
                noise = comfy.sample.prepare_noise(latent_image, current_seed, None)
                samples = comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler,
                    conditioning[0], conditioning[1], latent_image
                )

                decoded = vae.decode(samples)[0].cpu()  # [gen_h8, gen_w8, 3]

                # Store individual tile output using same extraction as blend_and_place_tile
                start_x = (overlap_x if has_left_neighbor else 0) if controlnet_active else 0
                start_y = (overlap_y if has_top_neighbor else 0) if controlnet_active else 0
                individual_tiles.append(
                    decoded[start_y:start_y + tile_height, start_x:start_x + tile_width, :].unsqueeze(0)
                )

                blend_and_place_tile(
                    final_tensor, decoded,
                    final_pos_x, final_pos_y,
                    tile_width, tile_height,
                    overlap_x, overlap_y,
                    has_left_neighbor, has_top_neighbor,
                    controlnet_active
                )
                pbar.update(1)
```

- [ ] **Step 4: Commit**

```
git add node.py
git commit -m "feat: expanded canvas generation + correct extraction in node.py"
```

---

### Task 4: Update `node_advanced.py`

**Files:**
- Modify: `node_advanced.py`

Same logic as Task 3, but uses guider/sampler/sigmas instead of `comfy.sample.sample`. `node_advanced.py` does not have `global_positive`/`global_negative` — it encodes only `current_prompt` and `""`.

- [ ] **Step 1: Update import**

```python
from .utils import parse_tile_prompts
from .utils import apply_controlnet_to_conditioning, blend_and_place_tile
from .utils import combine_guider_conditioning, restore_guider_conditioning
```

- [ ] **Step 2: Delete pre-loop gen_width/gen_height lines**

Remove:
```python
gen_width = ((tile_width + 7) // 8) * 8
gen_height = ((tile_height + 7) // 8) * 8
```

- [ ] **Step 3: Replace the full loop body**

```python
        for y in range(grid_height):
            for x in range(grid_width):
                idx = y * grid_width + x
                current_prompt = tile_prompts[idx]
                current_seed = seed + idx
                final_pos_x = x * tile_width
                final_pos_y = y * tile_height

                has_left_neighbor = x > 0
                has_top_neighbor = y > 0
                controlnet_active = controlnet is not None

                if controlnet_active:
                    gen_w = (tile_width
                             + (overlap_x if has_left_neighbor else 0)
                             + (overlap_x if seamlessX and x == grid_width - 1 else 0))
                    gen_h = (tile_height
                             + (overlap_y if has_top_neighbor else 0)
                             + (overlap_y if seamlessY and y == grid_height - 1 else 0))
                else:
                    gen_w, gen_h = tile_width, tile_height

                gen_w8 = ((gen_w + 7) // 8) * 8
                gen_h8 = ((gen_h + 7) // 8) * 8

                print(f"Tile ({x+1},{y+1}) gen canvas: {gen_w8}x{gen_h8}, canvas pos: ({final_pos_x},{final_pos_y})")
                print(f"Generating tile ({x+1},{y+1}) prompt: {current_prompt}")

                pos_cond = clip.encode_from_tokens_scheduled(clip.tokenize(current_prompt))
                neg_cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))

                if controlnet_active:
                    working_tensor = torch.zeros((1, gen_h8, gen_w8, 3), dtype=torch.float32)

                    if has_left_neighbor:
                        source_x = final_pos_x - overlap_x
                        source_start_y = final_pos_y
                        source_end_y = min(final_pos_y + tile_height, final_height)
                        source_height = source_end_y - source_start_y
                        target_start_y = overlap_y if has_top_neighbor else 0
                        copy_height = min(source_height, gen_h8 - target_start_y)
                        if copy_height > 0:
                            working_tensor[0,
                                target_start_y:target_start_y + copy_height, :overlap_x, :
                            ] = final_tensor[0,
                                source_start_y:source_start_y + copy_height,
                                source_x:source_x + overlap_x, :]

                    if seamlessX and x == grid_width - 1 and overlap_x > 0:
                        wrap_target_x = overlap_x + tile_width
                        source_start_y = final_pos_y
                        source_end_y = min(final_pos_y + tile_height, final_height)
                        source_height = source_end_y - source_start_y
                        target_start_y = overlap_y if has_top_neighbor else 0
                        copy_height = min(source_height, gen_h8 - target_start_y)
                        if copy_height > 0 and wrap_target_x + overlap_x <= gen_w8:
                            working_tensor[0,
                                target_start_y:target_start_y + copy_height,
                                wrap_target_x:wrap_target_x + overlap_x, :
                            ] = final_tensor[0,
                                source_start_y:source_start_y + copy_height, 0:overlap_x, :]

                    if seamlessY and y == grid_height - 1 and overlap_y > 0:
                        wrap_target_y = overlap_y + tile_height
                        source_start_x = final_pos_x
                        source_end_x = min(final_pos_x + tile_width, final_width)
                        source_width = source_end_x - source_start_x
                        target_start_x = overlap_x if has_left_neighbor else 0
                        copy_width = min(source_width, gen_w8 - target_start_x)
                        if copy_width > 0 and wrap_target_y + overlap_y <= gen_h8:
                            working_tensor[0,
                                wrap_target_y:wrap_target_y + overlap_y,
                                target_start_x:target_start_x + copy_width, :
                            ] = final_tensor[0,
                                0:overlap_y, source_start_x:source_start_x + copy_width, :]

                    if has_top_neighbor:
                        source_y = final_pos_y - overlap_y
                        source_start_x = final_pos_x
                        source_end_x = min(final_pos_x + tile_width, final_width)
                        source_width = source_end_x - source_start_x
                        target_start_x = overlap_x if has_left_neighbor else 0
                        copy_width = min(source_width, gen_w8 - target_start_x)
                        if copy_width > 0:
                            working_tensor[0,
                                :overlap_y, target_start_x:target_start_x + copy_width, :
                            ] = final_tensor[0,
                                source_y:source_y + overlap_y,
                                source_start_x:source_start_x + copy_width, :]

                    if has_left_neighbor and has_top_neighbor:
                        corner_source_x = final_pos_x - overlap_x
                        corner_source_y = final_pos_y - overlap_y
                        working_tensor[0, :overlap_y, :overlap_x, :] = final_tensor[0,
                            corner_source_y:corner_source_y + overlap_y,
                            corner_source_x:corner_source_x + overlap_x, :]

                    conditioning = apply_controlnet_to_conditioning(
                        positive=pos_cond, negative=neg_cond,
                        control_net=controlnet, image=working_tensor,
                        strength=controlnet_strength, start_percent=0.0, end_percent=1.0, vae=vae
                    )
                else:
                    conditioning = (pos_cond, neg_cond)

                latent_image = torch.zeros((1, 4, gen_h8 // 8, gen_w8 // 8), device=device)

                if noise is not None:
                    tile_noise = noise.generate_noise({"samples": latent_image})
                else:
                    tile_noise = comfy.sample.prepare_noise(latent_image, current_seed)

                guider.set_conds(conditioning[0], conditioning[1])
                samples = guider.sample(
                    tile_noise, latent_image, sampler, sigmas,
                    denoise_mask=None, disable_pbar=False, seed=current_seed
                )

                decoded = vae.decode(samples)[0].cpu()  # [gen_h8, gen_w8, 3]

                start_x = (overlap_x if has_left_neighbor else 0) if controlnet_active else 0
                start_y = (overlap_y if has_top_neighbor else 0) if controlnet_active else 0
                individual_tiles.append(
                    decoded[start_y:start_y + tile_height, start_x:start_x + tile_width, :].unsqueeze(0)
                )

                blend_and_place_tile(
                    final_tensor, decoded,
                    final_pos_x, final_pos_y,
                    tile_width, tile_height,
                    overlap_x, overlap_y,
                    has_left_neighbor, has_top_neighbor,
                    controlnet_active
                )
                pbar.update(1)
```

- [ ] **Step 4: Commit**

```
git add node_advanced.py
git commit -m "feat: expanded canvas generation + correct extraction in node_advanced.py"
```

---

## Self-Review Checklist

- **Spec Section 1 (Canvas Geometry):** Covered in Tasks 3 & 4 — gen_w/gen_h computed per-tile with controlnet_active guard. ✓
- **Spec Section 2 (Extraction):** `blend_and_place_tile` computes `start_x`/`start_y` and slices; `individual_tiles` uses same start offsets. ✓
- **Spec Section 3 (Blend):** `blend_and_place_tile` handles left, top, corner. ✓
- **Spec Section 4 (ControlNet-Optional):** `controlnet_active` flag gates all expansion, working_tensor construction, and blend. ✓
- **Spec Section 5 (Seamless):** `seamlessX`/`seamlessY` expands gen_w/gen_h and fills working_tensor wrap strips; post-loop trim unchanged. ✓
- **Resource Management:** No `vae.encode()` shape probe in loop; `decoded.cpu()` called immediately after decode; models not recreated per tile. ✓
- **Type consistency:** `blend_and_place_tile` signature in image_utils.py, __init__.py, node.py, node_advanced.py all match. ✓
