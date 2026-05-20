# Adaptive Tiled Image Detailer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `LLMAdaptiveTileDetailer` ComfyUI node that automatically scales per-tile denoise strength based on latent-space variance — flat tiles (sky, walls) get low denoise, detailed tiles (faces, fur) get high denoise.

**Architecture:** Two-pass approach: Pass 1 measures latent variance for every tile and normalizes it with a gamma curve into a per-tile denoise value. Pass 2 runs the standard sampling loop with those per-tile values and emits a pixel-space heatmap. Pure utility functions are extracted at module level for testability without ComfyUI.

**Tech Stack:** Python 3, PyTorch, ComfyUI node API (`comfy.sample`, `comfy.samplers`, `comfy.utils.ProgressBar`)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `node_detailer_adaptive.py` | Create | `LLMAdaptiveTileDetailer` class + four module-level pure functions |
| `__init__.py` | Modify | Register new node in the merged `NODE_CLASS_MAPPINGS` |
| `tests/test_adaptive_detailer.py` | Create | Unit tests for the four pure functions |

---

### Task 1: Create node skeleton and register in `__init__.py`

**Files:**
- Create: `node_detailer_adaptive.py`
- Modify: `__init__.py`

- [ ] **Step 1: Create `node_detailer_adaptive.py` with class stub**

```python
import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

from .utils import feather_blend_latent, _compute_center_grid


class LLMAdaptiveTileDetailer:

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
                "denoise_min": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_max": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "curve": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("refined_latent", "denoise_map")
    FUNCTION = "detail"
    CATEGORY = "image/generation"

    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler,
               denoise_min, denoise_max, curve, tile_size, overlap):
        raise NotImplementedError


NODE_CLASS_MAPPINGS = {
    "LLMAdaptiveTileDetailer": LLMAdaptiveTileDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMAdaptiveTileDetailer": "Adaptive Tiled Image Detailer",
}
```

- [ ] **Step 2: Register the new node in `__init__.py`**

The current `__init__.py` `else` block ends with:

```python
    from .node_detailer import NODE_CLASS_MAPPINGS as DET_NCM, NODE_DISPLAY_NAME_MAPPINGS as DET_NDCM

    # Combine all mappings
    NODE_CLASS_MAPPINGS = {**TILE_NCM, **ADV_NCM, **DET_NCM}
    NODE_DISPLAY_NAME_MAPPINGS = {**TILE_NDCM, **ADV_NDCM, **DET_NDCM}
```

Replace it with:

```python
    from .node_detailer import NODE_CLASS_MAPPINGS as DET_NCM, NODE_DISPLAY_NAME_MAPPINGS as DET_NDCM
    from .node_detailer_adaptive import NODE_CLASS_MAPPINGS as ADET_NCM, NODE_DISPLAY_NAME_MAPPINGS as ADET_NDCM

    # Combine all mappings
    NODE_CLASS_MAPPINGS = {**TILE_NCM, **ADV_NCM, **DET_NCM, **ADET_NCM}
    NODE_DISPLAY_NAME_MAPPINGS = {**TILE_NDCM, **ADV_NDCM, **DET_NDCM, **ADET_NDCM}
```

- [ ] **Step 3: Commit**

```bash
git add node_detailer_adaptive.py __init__.py
git commit -m "feat: add LLMAdaptiveTileDetailer skeleton and registration"
```

---

### Task 2: `_tile_variances` — test then implement

**Files:**
- Modify: `node_detailer_adaptive.py` (add module-level function before the class)
- Create: `tests/test_adaptive_detailer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_adaptive_detailer.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from node_detailer_adaptive import _tile_variances


def test_tile_variances_flat_returns_zero():
    canvas = torch.zeros(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_variances(canvas, coords)
    assert result == [pytest.approx(0.0)]


def test_tile_variances_nonzero_for_random():
    torch.manual_seed(42)
    canvas = torch.randn(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_variances(canvas, coords)
    assert result[0] > 0.0


def test_tile_variances_flat_less_than_varied():
    # Top-left quadrant is flat (zeros); bottom-right has a ramp (non-zero variance)
    canvas = torch.zeros(1, 4, 32, 32)
    ramp = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4).expand(1, 4, 4, 4).clone()
    canvas[:, :, 16:20, 16:20] = ramp
    coords = [(0, 0, 16, 16), (16, 16, 32, 32)]
    result = _tile_variances(canvas, coords)
    assert result[0] < result[1]
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_adaptive_detailer.py -v
```

Expected: `ImportError: cannot import name '_tile_variances' from 'node_detailer_adaptive'`

- [ ] **Step 3: Implement `_tile_variances` in `node_detailer_adaptive.py`**

Add this function before the `LLMAdaptiveTileDetailer` class (after the imports):

```python
def _tile_variances(canvas, tile_coords):
    """
    canvas: [B, C, H, W] latent tensor
    tile_coords: list of (y1, x1, y2, x2) in latent space
    Returns: list of float — per-tile spatial variance averaged across channels
    """
    return [
        canvas[:, :, y1:y2, x1:x2].var(dim=[2, 3]).mean().item()
        for (y1, x1, y2, x2) in tile_coords
    ]
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_adaptive_detailer.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add node_detailer_adaptive.py tests/test_adaptive_detailer.py
git commit -m "feat: add _tile_variances with tests"
```

---

### Task 3: `_variances_to_denoise` — test then implement

**Files:**
- Modify: `node_detailer_adaptive.py`
- Modify: `tests/test_adaptive_detailer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adaptive_detailer.py`:

```python
from node_detailer_adaptive import _variances_to_denoise


def test_uniform_variances_all_return_denoise_max():
    variances = [0.5, 0.5, 0.5]
    result = _variances_to_denoise(variances, curve=1.5, denoise_min=0.05, denoise_max=0.35)
    for t, denoise in result:
        assert t == pytest.approx(1.0)
        assert denoise == pytest.approx(0.35)


def test_linear_curve_maps_extremes_correctly():
    variances = [0.0, 1.0]
    result = _variances_to_denoise(variances, curve=1.0, denoise_min=0.05, denoise_max=0.35)
    t0, d0 = result[0]
    t1, d1 = result[1]
    assert t0 == pytest.approx(0.0)
    assert d0 == pytest.approx(0.05)   # flat → denoise_min
    assert t1 == pytest.approx(1.0)
    assert d1 == pytest.approx(0.35)   # detailed → denoise_max


def test_curve_gt_one_biases_midpoint_toward_min():
    variances = [0.0, 0.5, 1.0]
    linear = _variances_to_denoise(variances, curve=1.0, denoise_min=0.0, denoise_max=1.0)
    curved = _variances_to_denoise(variances, curve=2.0, denoise_min=0.0, denoise_max=1.0)
    # 0.5^1.0=0.5 vs 0.5^2.0=0.25 → curved midpoint gets lower denoise
    assert curved[1][1] < linear[1][1]


def test_single_variance_returns_denoise_max():
    # Single tile → v_min == v_max → t=1.0 → denoise_max
    result = _variances_to_denoise([0.7], curve=1.5, denoise_min=0.05, denoise_max=0.35)
    assert result[0][1] == pytest.approx(0.35)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_adaptive_detailer.py::test_uniform_variances_all_return_denoise_max -v
```

Expected: `ImportError: cannot import name '_variances_to_denoise'`

- [ ] **Step 3: Implement `_variances_to_denoise`**

Add after `_tile_variances` in `node_detailer_adaptive.py`:

```python
def _variances_to_denoise(variances, curve, denoise_min, denoise_max):
    """
    variances: list of float (one per tile)
    curve: gamma exponent; >1 biases most tiles toward denoise_min
    Returns: list of (t, denoise) tuples where
      t      — pre-curve normalized variance in [0,1] (used for heatmap)
      denoise — final per-tile denoise value
    """
    v_min = min(variances)
    v_max = max(variances)
    result = []
    for v in variances:
        t = 1.0 if v_max == v_min else (v - v_min) / (v_max - v_min)
        t_curved = t ** curve
        denoise = denoise_min + t_curved * (denoise_max - denoise_min)
        result.append((t, denoise))
    return result
```

- [ ] **Step 4: Run all tests**

```
pytest tests/test_adaptive_detailer.py -v
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add node_detailer_adaptive.py tests/test_adaptive_detailer.py
git commit -m "feat: add _variances_to_denoise with tests"
```

---

### Task 4: `_t_to_rgb` and `_build_denoise_map` — test then implement

**Files:**
- Modify: `node_detailer_adaptive.py`
- Modify: `tests/test_adaptive_detailer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adaptive_detailer.py`:

```python
from node_detailer_adaptive import _t_to_rgb, _build_denoise_map


def test_t_to_rgb_zero_is_blue():
    r, g, b = _t_to_rgb(0.0)
    assert r == pytest.approx(0.0, abs=1e-5)
    assert g == pytest.approx(0.0, abs=1e-5)
    assert b == pytest.approx(1.0, abs=1e-5)


def test_t_to_rgb_one_is_red():
    r, g, b = _t_to_rgb(1.0)
    assert r == pytest.approx(1.0, abs=1e-5)
    assert g == pytest.approx(0.0, abs=1e-5)
    assert b == pytest.approx(0.0, abs=1e-5)


def test_t_to_rgb_half_is_green():
    r, g, b = _t_to_rgb(0.5)
    assert r == pytest.approx(0.0, abs=1e-5)
    assert g == pytest.approx(1.0, abs=1e-5)
    assert b == pytest.approx(0.0, abs=1e-5)


def test_build_denoise_map_shape():
    coords = [(0, 0, 4, 4)]
    t_values = [0.5]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4)
    assert result.shape == (1, 32, 32, 3)


def test_build_denoise_map_blue_for_t_zero():
    coords = [(0, 0, 4, 4)]
    t_values = [0.0]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4)
    assert result[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)  # red=0
    assert result[0, 0, 0, 2].item() == pytest.approx(1.0, abs=1e-5)  # blue=1


def test_build_denoise_map_red_for_t_one():
    coords = [(0, 0, 4, 4)]
    t_values = [1.0]
    result = _build_denoise_map(coords, t_values, canvas_h=4, canvas_w=4)
    assert result[0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)  # red=1
    assert result[0, 0, 0, 2].item() == pytest.approx(0.0, abs=1e-5)  # blue=0


def test_build_denoise_map_two_tiles_distinct_colors():
    # top-left tile (t=0→blue), bottom-right tile (t=1→red)
    coords = [(0, 0, 4, 4), (4, 4, 8, 8)]
    t_values = [0.0, 1.0]
    result = _build_denoise_map(coords, t_values, canvas_h=8, canvas_w=8)
    # Top-left tile region — sample interior pixel (2*8, 2*8) = (16, 16)
    assert result[0, 16, 16, 2].item() == pytest.approx(1.0, abs=1e-5)  # blue
    # Bottom-right tile region — sample interior pixel (4*8+4, 4*8+4) = (36, 36)
    assert result[0, 36, 36, 0].item() == pytest.approx(1.0, abs=1e-5)  # red
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_adaptive_detailer.py::test_t_to_rgb_zero_is_blue -v
```

Expected: `ImportError: cannot import name '_t_to_rgb'`

- [ ] **Step 3: Implement `_t_to_rgb`**

Add after `_variances_to_denoise` in `node_detailer_adaptive.py`:

```python
def _t_to_rgb(t):
    """
    Map t∈[0,1] to RGB via HSV hue sweep (S=1, V=1).
    t=0 → blue (hue=240°), t=0.5 → green (hue=120°), t=1 → red (hue=0°).
    """
    hue = (1.0 - t) * 240.0   # degrees: 0=red, 120=green, 240=blue
    h = hue / 60.0
    i = int(h) % 6
    f = h - int(h)
    q = 1.0 - f
    segments = [
        (1.0, f,   0.0),  # 0: red→yellow
        (q,   1.0, 0.0),  # 1: yellow→green
        (0.0, 1.0, f  ),  # 2: green→cyan
        (0.0, q,   1.0),  # 3: cyan→blue
        (f,   0.0, 1.0),  # 4: blue→magenta
        (1.0, 0.0, q  ),  # 5: magenta→red
    ]
    return segments[i]
```

- [ ] **Step 4: Implement `_build_denoise_map`**

Add after `_t_to_rgb` in `node_detailer_adaptive.py`:

```python
def _build_denoise_map(tile_coords, t_values, canvas_h, canvas_w):
    """
    tile_coords: list of (y1, x1, y2, x2) in latent space
    t_values:    list of pre-curve normalized variance [0,1], one per tile
    canvas_h, canvas_w: latent-space dimensions (pixel dims = these × 8)
    Returns: IMAGE tensor [1, canvas_h*8, canvas_w*8, 3]
    """
    H, W = canvas_h * 8, canvas_w * 8
    img = torch.zeros(1, H, W, 3)
    for (y1, x1, y2, x2), t in zip(tile_coords, t_values):
        r, g, b = _t_to_rgb(t)
        img[0, y1 * 8:y2 * 8, x1 * 8:x2 * 8, 0] = r
        img[0, y1 * 8:y2 * 8, x1 * 8:x2 * 8, 1] = g
        img[0, y1 * 8:y2 * 8, x1 * 8:x2 * 8, 2] = b
    return img
```

- [ ] **Step 5: Run all tests**

```
pytest tests/test_adaptive_detailer.py -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add node_detailer_adaptive.py tests/test_adaptive_detailer.py
git commit -m "feat: add _t_to_rgb and _build_denoise_map with tests"
```

---

### Task 5: Implement `detail()` — wire the two passes

**Files:**
- Modify: `node_detailer_adaptive.py`

- [ ] **Step 1: Implement `detail()`**

Replace `raise NotImplementedError` in the `detail` method with:

```python
    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler,
               denoise_min, denoise_max, curve, tile_size, overlap):

        canvas = upscaled_latent["samples"].clone()
        _, _, H, W = canvas.shape

        tile_l = tile_size // 8
        overlap_l = overlap // 8
        if overlap_l >= tile_l:
            overlap_l = tile_l // 2
            print(f"[LLMAdaptiveTileDetailer] Warning: overlap clamped to "
                  f"{overlap_l * 8}px (overlap must be < tile_size)")

        cols, rows, start_x, start_y = _compute_center_grid(W, H, tile_l, overlap_l)
        stride = tile_l - overlap_l

        print(f"[LLMAdaptiveTileDetailer] Latent {W}x{H} | "
              f"tile_l={tile_l} overlap_l={overlap_l} stride={stride} | "
              f"grid cols={cols} rows={rows} ({(rows+1)*(cols+1)} tiles)")

        # --- Pass 1: collect valid tile coords and measure variance ---
        tile_coords = []
        for r in range(rows + 1):
            for c in range(cols + 1):
                y1 = max(0, start_y + r * stride)
                x1 = max(0, start_x + c * stride)
                y2 = min(H, y1 + tile_l)
                x2 = min(W, x1 + tile_l)
                if y2 > y1 and x2 > x1:
                    tile_coords.append((y1, x1, y2, x2))

        variances = _tile_variances(canvas, tile_coords)
        td_pairs = _variances_to_denoise(variances, curve, denoise_min, denoise_max)
        denoise_map_img = _build_denoise_map(tile_coords, [t for t, _ in td_pairs], H, W)

        # --- Pass 2: sample each tile with its computed denoise ---
        pbar = ProgressBar(len(tile_coords))
        tile_idx = 0
        for r in range(rows + 1):
            for c in range(cols + 1):
                y1 = max(0, start_y + r * stride)
                x1 = max(0, start_x + c * stride)
                y2 = min(H, y1 + tile_l)
                x2 = min(W, x1 + tile_l)
                if y2 <= y1 or x2 <= x1:
                    pbar.update(1)
                    continue

                t_val, tile_denoise = td_pairs[tile_idx]
                var = variances[tile_idx]
                tile_idx += 1

                print(f"[LLMAdaptiveTileDetailer] tile ({r},{c}) "
                      f"var={var:.4f} t={t_val:.2f} denoise={tile_denoise:.3f}")

                tile_seed = seed + r * (cols + 1) + c
                tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

                noise = comfy.sample.prepare_noise(tile_latent, tile_seed, None)
                refined = comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler,
                    positive, negative, tile_latent,
                    denoise=tile_denoise,
                )

                feather_blend_latent(
                    canvas, refined, y1, x1, overlap_l,
                    has_left=(x1 > 0),
                    has_top=(y1 > 0),
                )

                comfy.model_management.soft_empty_cache()
                pbar.update(1)

        return ({"samples": canvas}, denoise_map_img)
```

- [ ] **Step 2: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests pass (no regressions in `test_detailer.py`, `test_blend_and_place.py`, `test_mask_resize.py`)

- [ ] **Step 3: Commit**

```bash
git add node_detailer_adaptive.py
git commit -m "feat: implement LLMAdaptiveTileDetailer two-pass detail() method"
```
