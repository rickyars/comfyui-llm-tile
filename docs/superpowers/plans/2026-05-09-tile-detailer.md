# LLM Tile Sequential Detailer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `LLMTileSequentialDetailer`, a ComfyUI node that applies partial denoising tile-by-tile over an upscaled latent to inject fine detail, replacing Tiled Diffusion to eliminate VRAM saturation (~40 s/it → ~1.2 it/s).

**Architecture:** A new `node_detailer.py` holds the node class only. Pure helper functions (`feather_blend_latent`, `_compute_center_grid`) live in `utils/image_utils.py` so they are importable in tests without hitting package-relative imports. Tiles are processed sequentially; `comfy.model_management.soft_empty_cache()` runs between each tile. Note: `_compute_center_grid` is placed in `utils/image_utils.py` rather than `node_detailer.py` (spec deviation) because tests cannot import files that use relative imports — confirmed by the existing test pattern in `tests/test_blend_and_place.py`.

**Tech Stack:** PyTorch, ComfyUI (`comfy.sample`, `comfy.model_management`, `comfy.samplers`, `comfy.utils.ProgressBar`)

---

## File Map

| File | Change |
|---|---|
| `utils/image_utils.py` | Add `feather_blend_latent()` and `_compute_center_grid()` |
| `utils/__init__.py` | Export both new functions |
| `node_detailer.py` | New — `LLMTileSequentialDetailer` class |
| `__init__.py` | Import and merge `node_detailer` mappings |
| `tests/test_detailer.py` | New — unit tests for blend and grid math |

---

### Task 1: `feather_blend_latent` and `_compute_center_grid` utilities + tests

**Files:**
- Modify: `utils/image_utils.py`
- Create: `tests/test_detailer.py`

- [ ] **Step 1: Create `tests/test_detailer.py` with three failing tests**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from utils.image_utils import feather_blend_latent, _compute_center_grid


def test_feather_blend_latent_left_edge():
    overlap_l = 4
    # Previously placed tile occupies x=0..7 with value 1.0; rest 0.0
    canvas = torch.zeros(1, 4, 8, 16)
    canvas[:, :, :, 0:8] = 1.0

    # New tile placed at x1=4; tile covers x=4..11 in canvas; value 0.5
    refined = torch.full((1, 4, 8, 8), 0.5)

    feather_blend_latent(canvas, refined, y1=0, x1=4, overlap_l=overlap_l,
                         has_left=True, has_top=False)

    # Left edge of overlap (x=4): alpha=0 → old canvas value preserved
    assert canvas[0, 0, 0, 4].item() == pytest.approx(1.0, abs=1e-5)
    # Right edge of overlap (x=7): alpha=1 → full refined value
    assert canvas[0, 0, 0, 7].item() == pytest.approx(0.5, abs=1e-5)
    # Interior beyond overlap (x=8): hard-set to refined
    assert canvas[0, 0, 0, 8].item() == pytest.approx(0.5, abs=1e-5)
    # Ramp is monotonically non-increasing across the overlap zone
    vals = [canvas[0, 0, 0, 4 + i].item() for i in range(4)]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


def test_feather_blend_latent_top_edge():
    overlap_l = 4
    # Previously placed tile occupies y=0..7 with value 1.0
    canvas = torch.zeros(1, 4, 16, 8)
    canvas[:, :, 0:8, :] = 1.0

    # New tile placed at y1=4; tile covers y=4..11; value 0.5
    refined = torch.full((1, 4, 8, 8), 0.5)

    feather_blend_latent(canvas, refined, y1=4, x1=0, overlap_l=overlap_l,
                         has_left=False, has_top=True)

    # Top edge of overlap (y=4): alpha=0 → old canvas value preserved
    assert canvas[0, 0, 4, 0].item() == pytest.approx(1.0, abs=1e-5)
    # Bottom edge of overlap (y=7): alpha=1 → full refined value
    assert canvas[0, 0, 7, 0].item() == pytest.approx(0.5, abs=1e-5)
    # Interior beyond overlap (y=8): hard-set to refined
    assert canvas[0, 0, 8, 0].item() == pytest.approx(0.5, abs=1e-5)
    # Ramp is monotonically non-increasing across the overlap zone
    vals = [canvas[0, 0, 4 + i, 0].item() for i in range(4)]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


def test_center_anchored_grid():
    # 1000x2048 portrait: latent W=125, H=256
    # tile_size=1024 → tile_l=128; overlap=64 → overlap_l=8
    cols, rows, start_x, start_y = _compute_center_grid(W=125, H=256, tile_l=128, overlap_l=8)

    assert cols == 1
    assert rows == 2
    # Image (125) is narrower than tile_l (128); centering goes slightly negative
    assert start_x == -2
    # 4 latent pixels of top/bottom padding: (256 - 248) // 2 = 4
    assert start_y == 4

    # All clamped tile coordinates must stay within latent bounds
    stride = 128 - 8  # 120
    for r in range(rows + 1):
        for c in range(cols + 1):
            y1 = max(0, start_y + r * stride)
            x1 = max(0, start_x + c * stride)
            y2 = min(256, y1 + 128)
            x2 = min(125, x1 + 128)
            assert 0 <= y1 <= 256
            assert 0 <= x1 <= 125
            assert y1 <= y2 <= 256
            assert x1 <= x2 <= 125
```

- [ ] **Step 2: Run tests to verify they fail**

Run from the project root (`E:\StableDiffusion\ComfyUI\custom_nodes\comfyui-llm-tile`):

```
pytest tests/test_detailer.py -v
```

Expected: `ImportError: cannot import name 'feather_blend_latent' from 'utils.image_utils'`

- [ ] **Step 3: Add `feather_blend_latent` and `_compute_center_grid` to `utils/image_utils.py`**

Append both functions at the bottom of `utils/image_utils.py` (after the existing `gaussian_blend_tiles` function):

```python
def feather_blend_latent(canvas, refined, y1, x1, overlap_l, has_left, has_top):
    """
    Write a refined latent tile into canvas with linear feathering on overlap edges.

    canvas:   [B, C, H, W] CPU tensor being assembled in-place
    refined:  [B, C, tile_h, tile_w] sampler output (moved to CPU internally)
    y1, x1:   top-left insertion corner in canvas coordinates
    overlap_l: overlap width/height in latent pixels
    has_left:  True when a previously placed tile overlaps from the left
    has_top:   True when a previously placed tile overlaps from above
    """
    _, _, tile_h, tile_w = refined.shape
    refined_cpu = refined.cpu()

    # Save existing canvas values in overlap zones before overwriting
    left_zone = (canvas[:, :, y1:y1 + tile_h, x1:x1 + overlap_l].clone()
                 if (has_left and overlap_l > 0) else None)
    top_zone = (canvas[:, :, y1:y1 + overlap_l, x1:x1 + tile_w].clone()
                if (has_top and overlap_l > 0) else None)
    corner_zone = (canvas[:, :, y1:y1 + overlap_l, x1:x1 + overlap_l].clone()
                   if (has_left and has_top and overlap_l > 0) else None)

    # Hard-write the full refined tile
    canvas[:, :, y1:y1 + tile_h, x1:x1 + tile_w] = refined_cpu

    # Left overlap: ramp alpha 0→1 across overlap columns (old canvas → refined)
    if left_zone is not None and tile_w > overlap_l:
        alpha = torch.linspace(0.0, 1.0, overlap_l).view(1, 1, 1, overlap_l)
        canvas[:, :, y1:y1 + tile_h, x1:x1 + overlap_l] = (
            (1.0 - alpha) * left_zone + alpha * refined_cpu[:, :, :, :overlap_l]
        )

    # Top overlap: ramp alpha 0→1 across overlap rows (old canvas → refined)
    if top_zone is not None and tile_h > overlap_l:
        alpha = torch.linspace(0.0, 1.0, overlap_l).view(1, 1, overlap_l, 1)
        canvas[:, :, y1:y1 + overlap_l, x1:x1 + tile_w] = (
            (1.0 - alpha) * top_zone + alpha * refined_cpu[:, :, :overlap_l, :]
        )

    # Corner: min(alpha_x, alpha_y) for smooth 2D diagonal blend
    if corner_zone is not None and tile_w > overlap_l and tile_h > overlap_l:
        alpha_x = torch.linspace(0.0, 1.0, overlap_l).view(1, 1, 1, overlap_l)
        alpha_y = torch.linspace(0.0, 1.0, overlap_l).view(1, 1, overlap_l, 1)
        alpha = torch.min(
            alpha_x.expand(1, 1, overlap_l, overlap_l),
            alpha_y.expand(1, 1, overlap_l, overlap_l),
        )
        canvas[:, :, y1:y1 + overlap_l, x1:x1 + overlap_l] = (
            (1.0 - alpha) * corner_zone + alpha * refined_cpu[:, :, :overlap_l, :overlap_l]
        )


def _compute_center_grid(W, H, tile_l, overlap_l):
    """
    Compute a center-anchored tile grid for a latent of size W x H.

    Returns (cols, rows, start_x, start_y). All values in latent-space pixels.
    start_x / start_y may be negative when the image is narrower/shorter than
    tile_l — clamp with max(0, ...) before use.
    """
    stride = tile_l - overlap_l
    cols = max(1, (W - overlap_l) // stride)
    rows = max(1, (H - overlap_l) // stride)
    start_x = (W - (cols * stride + overlap_l)) // 2
    start_y = (H - (rows * stride + overlap_l)) // 2
    return cols, rows, start_x, start_y
```

- [ ] **Step 4: Run all three tests to verify they pass**

```
pytest tests/test_detailer.py -v
```

Expected:
```
PASSED tests/test_detailer.py::test_feather_blend_latent_left_edge
PASSED tests/test_detailer.py::test_feather_blend_latent_top_edge
PASSED tests/test_detailer.py::test_center_anchored_grid
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add utils/image_utils.py tests/test_detailer.py
git commit -m "feat: add feather_blend_latent and _compute_center_grid utilities"
```

---

### Task 2: `node_detailer.py` — node class

**Files:**
- Create: `node_detailer.py`

- [ ] **Step 1: Create `node_detailer.py`**

```python
import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

from .utils import feather_blend_latent, _compute_center_grid


class LLMTileSequentialDetailer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "upscaled_latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.01}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("refined_latent",)
    FUNCTION = "detail"
    CATEGORY = "sampling/detailers"

    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler, denoise, tile_size, overlap):

        canvas = upscaled_latent["samples"].clone()
        _, _, H, W = canvas.shape

        tile_l = tile_size // 8
        overlap_l = overlap // 8
        if overlap_l >= tile_l:
            overlap_l = tile_l // 2
            print(f"[LLMTileSequentialDetailer] Warning: overlap clamped to "
                  f"{overlap_l * 8}px (must be < tile_size // 2)")

        cols, rows, start_x, start_y = _compute_center_grid(W, H, tile_l, overlap_l)
        stride = tile_l - overlap_l
        total_tiles = (rows + 1) * (cols + 1)

        print(f"[LLMTileSequentialDetailer] Latent {W}x{H} | "
              f"tile_l={tile_l} overlap_l={overlap_l} stride={stride} | "
              f"grid cols={cols} rows={rows} ({total_tiles} tiles)")

        pbar = ProgressBar(total_tiles)

        for r in range(rows + 1):
            for c in range(cols + 1):
                y1 = max(0, start_y + r * stride)
                x1 = max(0, start_x + c * stride)
                y2 = min(H, y1 + tile_l)
                x2 = min(W, x1 + tile_l)

                if y2 <= y1 or x2 <= x1:
                    pbar.update(1)
                    continue

                tile_seed = seed + r * (cols + 1) + c
                tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

                noise = comfy.sample.prepare_noise(tile_latent, tile_seed, None)
                refined = comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler,
                    positive, negative, tile_latent,
                    denoise=denoise,
                )

                feather_blend_latent(
                    canvas, refined, y1, x1, overlap_l,
                    has_left=(x1 > 0),
                    has_top=(y1 > 0),
                )

                comfy.model_management.soft_empty_cache()
                pbar.update(1)

        return ({"samples": canvas},)


NODE_CLASS_MAPPINGS = {
    "LLMTileSequentialDetailer": LLMTileSequentialDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMTileSequentialDetailer": "LLM Tile Sequential Detailer",
}
```

- [ ] **Step 2: Run the full test suite to confirm the new file doesn't break anything**

```
pytest tests/ -v
```

Expected: All previously passing tests still pass; `test_detailer.py` still shows 3 passed. (The node class itself cannot be unit-tested without a live ComfyUI model — that is expected.)

- [ ] **Step 3: Commit**

```bash
git add node_detailer.py
git commit -m "feat: add LLMTileSequentialDetailer node"
```

---

### Task 3: Wire up `__init__` files

**Files:**
- Modify: `utils/__init__.py`
- Modify: `__init__.py`

- [ ] **Step 1: Export new utilities from `utils/__init__.py`**

Current line in `utils/__init__.py`:
```python
from .image_utils import gaussian_blend_tiles, resize_mask_to_latent, blend_and_place_tile
```

Replace with:
```python
from .image_utils import gaussian_blend_tiles, resize_mask_to_latent, blend_and_place_tile, feather_blend_latent, _compute_center_grid
```

- [ ] **Step 2: Import and merge `node_detailer` mappings in `__init__.py`**

Current `__init__.py` (lines 4–10):
```python
from .node import NODE_CLASS_MAPPINGS as TILE_NCM, NODE_DISPLAY_NAME_MAPPINGS as TILE_NDCM
# Import your new advanced node mappings
from .node_advanced import NODE_CLASS_MAPPINGS as ADV_NCM, NODE_DISPLAY_NAME_MAPPINGS as ADV_NDCM

# Combine all mappings
NODE_CLASS_MAPPINGS = {**TILE_NCM, **ADV_NCM}
NODE_DISPLAY_NAME_MAPPINGS = {**TILE_NDCM, **ADV_NDCM}
```

Replace with:
```python
from .node import NODE_CLASS_MAPPINGS as TILE_NCM, NODE_DISPLAY_NAME_MAPPINGS as TILE_NDCM
from .node_advanced import NODE_CLASS_MAPPINGS as ADV_NCM, NODE_DISPLAY_NAME_MAPPINGS as ADV_NDCM
from .node_detailer import NODE_CLASS_MAPPINGS as DET_NCM, NODE_DISPLAY_NAME_MAPPINGS as DET_NDCM

NODE_CLASS_MAPPINGS = {**TILE_NCM, **ADV_NCM, **DET_NCM}
NODE_DISPLAY_NAME_MAPPINGS = {**TILE_NDCM, **ADV_NDCM, **DET_NDCM}
```

- [ ] **Step 3: Run the full test suite one final time**

```
pytest tests/ -v
```

Expected: All tests pass. No regressions.

- [ ] **Step 4: Commit**

```bash
git add utils/__init__.py __init__.py
git commit -m "feat: register LLMTileSequentialDetailer and export new utilities"
```

---

## Self-Review

**Spec coverage:**
- ✅ `feather_blend_latent` in `utils/image_utils.py` — Task 1
- ✅ `_compute_center_grid` — Task 1 (moved to `utils/image_utils.py` from spec's `node_detailer.py` for testability — justified above)
- ✅ `LLMTileSequentialDetailer` node class — Task 2
- ✅ CONDITIONING inputs (not text strings) — Task 2
- ✅ LATENT output — Task 2
- ✅ Center-anchored grid math — Tasks 1 + 2
- ✅ Zero-area tile skip — Task 2
- ✅ Overlap clamp guard with warning — Task 2
- ✅ `soft_empty_cache()` between tiles — Task 2
- ✅ `ProgressBar` — Task 2
- ✅ Per-tile seed offset (row-major) — Task 2
- ✅ Input clone (no mutation of original latent) — Task 2
- ✅ Category `sampling/detailers` — Task 2
- ✅ `utils/__init__.py` export — Task 3
- ✅ `__init__.py` merge — Task 3
- ✅ Unit tests: left edge blend, top edge blend, grid math — Task 1

**Placeholder scan:** No TBD, TODO, or vague steps. Every code step contains complete, runnable code.

**Type consistency:**
- `feather_blend_latent(canvas, refined, y1, x1, overlap_l, has_left, has_top)` — defined in Task 1, called identically in Task 2. ✓
- `_compute_center_grid(W, H, tile_l, overlap_l)` — defined in Task 1, called identically in Task 2. ✓
- `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` — defined in Task 2, merged in Task 3. ✓
