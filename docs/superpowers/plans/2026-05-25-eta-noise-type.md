# Eta Control & Noise Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-tile eta (SDE noise injection) and initial noise type selection to both tile detailer nodes.

**Architecture:** Switch both nodes from `comfy.sample.sample()` to `comfy.sample.sample_custom()` with manually-built samplers and sigmas. Eta is injected into the sampler via `extra_options` when the sampler supports it (detected by runtime signature inspection). Noise type dispatches to RES4LYF's `NOISE_GENERATOR_CLASSES_SIMPLE` when available, falling back to standard Gaussian. In the adaptive detailer, eta scales linearly with per-tile denoise so minimum-denoise tiles run ODE (eta=0) and maximum-denoise tiles run full SDE.

**Tech Stack:** PyTorch, ComfyUI (`comfy.samplers`, `comfy.sample`), RES4LYF (`beta/noise_classes.py`, optional dependency), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `utils/sampling_utils.py` | Create | Noise generation, sampler construction, eta support check |
| `utils/__init__.py` | Modify | Re-export new utilities |
| `conftest.py` | Modify | Add stubs for `sample_custom`, `ksampler`, `k_diffusion_sampling`, `calculate_sigmas` |
| `tests/test_sampling_utils.py` | Create | Unit tests for new utilities |
| `node_detailer.py` | Modify | New inputs, `sample_custom` path |
| `node_detailer_adaptive.py` | Modify | New inputs, per-tile eta scaling, `sample_custom` path |

---

## Task 1: Expand conftest.py stubs

The existing conftest stubs `comfy.samplers` as a bare module. New code needs `ksampler`, `k_diffusion_sampling`, `calculate_sigmas`, `sample_custom`, `prepare_noise`, and `soft_empty_cache`.

**Files:**
- Modify: `conftest.py`

- [ ] **Step 1: Replace `_make_comfy_stubs` with the expanded version**

Open `conftest.py` and replace the entire `_make_comfy_stubs` function with:

```python
def _make_comfy_stubs():
    import torch

    comfy = ModuleType('comfy')
    comfy.sample = ModuleType('comfy.sample')
    comfy.model_management = ModuleType('comfy.model_management')
    comfy.samplers = ModuleType('comfy.samplers')
    comfy.utils = ModuleType('comfy.utils')

    # KSampler stub
    ksample = ModuleType('comfy.samplers.KSampler')
    ksample.SAMPLERS = []
    ksample.SCHEDULERS = []
    comfy.samplers.KSampler = ksample

    # ProgressBar stub
    comfy.utils.ProgressBar = unittest.mock.MagicMock

    # prepare_noise — returns zeros matching the input shape
    comfy.sample.prepare_noise = unittest.mock.MagicMock(
        side_effect=lambda latent, seed, inds: torch.zeros_like(latent)
    )

    # sample_custom — returns a clone of latent_image (arg index 7)
    comfy.sample.sample_custom = unittest.mock.MagicMock(
        side_effect=lambda model, noise, cfg, sampler, sigmas, pos, neg, latent, **kw: latent.clone()
    )

    # k_diffusion_sampling stub — two functions: one with eta, one without
    k_diff = ModuleType('comfy.samplers.k_diffusion_sampling')
    def _sample_with_eta(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0):
        pass
    def _sample_without_eta(model, x, sigmas, extra_args=None, callback=None, disable=None):
        pass
    k_diff.sample_euler_ancestral = _sample_with_eta
    k_diff.sample_euler = _sample_without_eta
    comfy.samplers.k_diffusion_sampling = k_diff
    sys.modules['comfy.samplers.k_diffusion_sampling'] = k_diff

    # ksampler — returns a mock whose extra_options matches what was passed
    def _mock_ksampler(name, extra_options=None):
        m = unittest.mock.MagicMock()
        m.extra_options = dict(extra_options) if extra_options else {}
        return m
    comfy.samplers.ksampler = _mock_ksampler

    # calculate_sigmas — returns a 21-element descending tensor (20 steps + endpoint)
    comfy.samplers.calculate_sigmas = unittest.mock.MagicMock(
        return_value=torch.linspace(14.6, 0.0, 21)
    )

    # model_management utilities
    comfy.model_management.soft_empty_cache = unittest.mock.MagicMock()
    comfy.model_management.intermediate_device = unittest.mock.MagicMock(
        return_value=torch.device('cpu')
    )
    comfy.model_management.intermediate_dtype = unittest.mock.MagicMock(
        return_value=torch.float32
    )

    sys.modules['comfy'] = comfy
    sys.modules['comfy.sample'] = comfy.sample
    sys.modules['comfy.model_management'] = comfy.model_management
    sys.modules['comfy.samplers'] = comfy.samplers
    sys.modules['comfy.utils'] = comfy.utils
```

- [ ] **Step 2: Run all existing tests to confirm no regressions**

```
pytest tests/ -v
```

Expected: all existing tests PASS (the new stubs are additive only).

- [ ] **Step 3: Commit**

```bash
git add conftest.py
git commit -m "test: expand comfy stubs for sample_custom, ksampler, and k_diffusion_sampling"
```

---

## Task 2: Create `utils/sampling_utils.py` with tests

**Files:**
- Create: `tests/test_sampling_utils.py`
- Create: `utils/sampling_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sampling_utils.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from utils.sampling_utils import check_eta_support, prepare_noise_typed, build_tile_sampler


# --- check_eta_support ---

def test_check_eta_support_true_for_sampler_with_eta():
    # conftest stubs k_diffusion_sampling.sample_euler_ancestral with eta param
    assert check_eta_support("euler_ancestral") is True


def test_check_eta_support_false_for_sampler_without_eta():
    # conftest stubs k_diffusion_sampling.sample_euler without eta param
    assert check_eta_support("euler") is False


def test_check_eta_support_false_for_unknown_sampler():
    assert check_eta_support("nonexistent_sampler_xyz") is False


# --- prepare_noise_typed ---

def test_prepare_noise_typed_gaussian_delegates_to_comfy():
    import comfy.sample
    tile = torch.zeros(1, 4, 8, 8)
    comfy.sample.prepare_noise.reset_mock()
    prepare_noise_typed(tile, seed=42, noise_type="gaussian", sigma_min=0.03, sigma_max=14.6)
    comfy.sample.prepare_noise.assert_called_once_with(tile, 42, None)


def test_prepare_noise_typed_returns_tensor_matching_input_shape():
    tile = torch.zeros(1, 4, 8, 8)
    result = prepare_noise_typed(tile, seed=42, noise_type="gaussian", sigma_min=0.03, sigma_max=14.6)
    assert isinstance(result, torch.Tensor)
    assert result.shape == tile.shape


def test_prepare_noise_typed_unknown_type_falls_back_to_gaussian():
    # With no RES4LYF present in test env, any non-gaussian type falls back
    import comfy.sample
    tile = torch.zeros(1, 4, 8, 8)
    comfy.sample.prepare_noise.reset_mock()
    prepare_noise_typed(tile, seed=0, noise_type="brownian", sigma_min=0.03, sigma_max=14.6)
    # Should call comfy fallback (RES4LYF not available in test env)
    comfy.sample.prepare_noise.assert_called_once()


# --- build_tile_sampler ---

def test_build_tile_sampler_includes_eta_when_supported():
    sampler = build_tile_sampler("euler_ancestral", tile_eta=0.7, eta_supported=True)
    assert sampler.extra_options == {"eta": 0.7}


def test_build_tile_sampler_omits_eta_when_unsupported():
    sampler = build_tile_sampler("euler", tile_eta=0.7, eta_supported=False)
    assert sampler.extra_options == {}


def test_build_tile_sampler_eta_zero_still_passes_when_supported():
    sampler = build_tile_sampler("euler_ancestral", tile_eta=0.0, eta_supported=True)
    assert sampler.extra_options == {"eta": 0.0}
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_sampling_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'utils.sampling_utils'`

- [ ] **Step 3: Create `utils/sampling_utils.py`**

```python
import inspect
import comfy.sample
import comfy.samplers

try:
    from RES4LYF.beta.noise_classes import (
        NOISE_GENERATOR_CLASSES_SIMPLE,
        NOISE_GENERATOR_NAMES_SIMPLE,
    )
    _RES4LYF_NOISE = True
except ImportError:
    NOISE_GENERATOR_NAMES_SIMPLE = ("gaussian",)
    _RES4LYF_NOISE = False


def check_eta_support(sampler_name):
    """Return True if the named sampler function accepts an eta keyword argument."""
    fn = getattr(comfy.samplers.k_diffusion_sampling, f"sample_{sampler_name}", None)
    if fn is None:
        return False
    return 'eta' in inspect.signature(fn).parameters


def prepare_noise_typed(tile_latent, seed, noise_type, sigma_min, sigma_max):
    """Generate initial tile noise using the specified distribution.

    Falls back to standard Gaussian if RES4LYF is not installed or noise_type is 'gaussian'.
    """
    if not _RES4LYF_NOISE or noise_type == "gaussian":
        return comfy.sample.prepare_noise(tile_latent, seed, None)
    cls = NOISE_GENERATOR_CLASSES_SIMPLE[noise_type]
    gen = cls(x=tile_latent, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    return gen(sigma=float(sigma_max), sigma_next=float(sigma_min))


def build_tile_sampler(sampler_name, tile_eta, eta_supported):
    """Build a KSAMPLER for one tile, injecting eta into extra_options if supported."""
    extra_opts = {'eta': tile_eta} if eta_supported else {}
    return comfy.samplers.ksampler(sampler_name, extra_options=extra_opts)
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_sampling_utils.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/sampling_utils.py tests/test_sampling_utils.py
git commit -m "feat: add sampling_utils with noise type dispatch and eta-aware sampler builder"
```

---

## Task 3: Export new utilities from `utils/__init__.py`

**Files:**
- Modify: `utils/__init__.py`

- [ ] **Step 1: Add exports**

Append to `utils/__init__.py`:

```python
from .sampling_utils import check_eta_support, prepare_noise_typed, build_tile_sampler, NOISE_GENERATOR_NAMES_SIMPLE
```

- [ ] **Step 2: Run all tests to confirm no regressions**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add utils/__init__.py
git commit -m "chore: re-export sampling_utils from utils package"
```

---

## Task 4: Update `node_detailer.py`

**Files:**
- Modify: `node_detailer.py`
- Modify: `tests/test_detailer.py` (add new tests)

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_detailer.py`:

```python
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
```

- [ ] **Step 2: Run new tests to confirm they fail**

```
pytest tests/test_detailer.py::test_detail_uses_sample_custom_not_sample tests/test_detailer.py::test_detail_eta_nonzero_passes_eta_to_ksampler tests/test_detailer.py::test_detail_input_types_include_noise_type_and_eta -v
```

Expected: FAIL (ImportError or AttributeError since `noise_type`/`eta` not yet in node).

- [ ] **Step 3: Update `node_detailer.py`**

Replace the entire file with:

```python
import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

from .utils import (
    feather_blend_latent, _compute_center_grid, _compute_tile_coords,
    check_eta_support, prepare_noise_typed, build_tile_sampler,
    NOISE_GENERATOR_NAMES_SIMPLE,
)


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
                "crop_to_tiles": ("BOOLEAN", {"default": False}),
                "noise_type": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                  "tooltip": "SDE noise injection per step. 0 = deterministic ODE. Only applies to samplers that support eta (e.g. euler_ancestral, rk_beta)."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("refined_latent",)
    FUNCTION = "detail"
    CATEGORY = "image/generation"

    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               tile_size, overlap, crop_to_tiles, noise_type, eta):

        canvas = upscaled_latent["samples"].clone()
        _, _, H, W = canvas.shape

        tile_l = tile_size // 8
        overlap_l = overlap // 8
        if overlap_l >= tile_l:
            overlap_l = tile_l // 2
            print(f"[LLMTileSequentialDetailer] Warning: overlap clamped to "
                  f"{overlap_l * 8}px (overlap must be < tile_size)")

        cols, rows = _compute_center_grid(W, H, tile_l, overlap_l)
        stride = tile_l - overlap_l
        total_tiles = (rows + 1) * (cols + 1)

        print(f"[LLMTileSequentialDetailer] Latent {W}x{H} | "
              f"tile_l={tile_l} overlap_l={overlap_l} stride={stride} | "
              f"grid cols={cols} rows={rows} ({total_tiles} tiles)")

        model_sampling = model.get_model_object("model_sampling")
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)
        eta_supported = check_eta_support(sampler_name)
        if eta > 0.0 and not eta_supported:
            print(f"[LLMTileSequentialDetailer] Warning: '{sampler_name}' does not support "
                  f"eta; eta will be ignored. Use 'rk_beta' for full RES4LYF eta control.")

        if denoise >= 1.0:
            tile_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        else:
            new_steps = int(steps / denoise)
            tile_sigmas = comfy.samplers.calculate_sigmas(
                model_sampling, scheduler, new_steps)[-(steps + 1):]
        tile_sigmas = tile_sigmas.to(model.load_device)
        sampler = build_tile_sampler(sampler_name, eta, eta_supported)

        pbar = ProgressBar(total_tiles)
        tile_coords = _compute_tile_coords(W, H, tile_l, cols, rows, overlap_l)
        n_cols = cols + 1

        for tile_idx, (y1, x1, y2, x2) in enumerate(tile_coords):
            r = tile_idx // n_cols
            c = tile_idx % n_cols
            tile_seed = seed + tile_idx
            tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

            noise = prepare_noise_typed(tile_latent, tile_seed, noise_type, sigma_min, sigma_max)
            refined = comfy.sample.sample_custom(
                model, noise, cfg, sampler, tile_sigmas,
                positive, negative, tile_latent,
            )

            feather_blend_latent(
                canvas, refined, y1, x1, overlap_l,
                has_left=(c > 0 and x1 < tile_coords[tile_idx - 1][3]),
                has_top=(r > 0 and y1 < tile_coords[tile_idx - n_cols][2]),
            )

            comfy.model_management.soft_empty_cache()
            pbar.update(1)

        if crop_to_tiles:
            y1_c, x1_c = tile_coords[0][0], tile_coords[0][1]
            y2_c, x2_c = tile_coords[-1][2], tile_coords[cols][3]
            canvas = canvas[:, :, y1_c:y2_c, x1_c:x2_c]

        return ({"samples": canvas},)


NODE_CLASS_MAPPINGS = {
    "LLMTileSequentialDetailer": LLMTileSequentialDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMTileSequentialDetailer": "Tiled Image Detailer",
}
```

- [ ] **Step 4: Run all tests**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add node_detailer.py tests/test_detailer.py
git commit -m "feat: add eta and noise_type inputs to LLMTileSequentialDetailer"
```

---

## Task 5: Update `node_detailer_adaptive.py`

**Files:**
- Modify: `node_detailer_adaptive.py`
- Modify: `tests/test_adaptive_detailer.py` (add new tests)

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_adaptive_detailer.py`:

```python
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


def test_adaptive_eta_scaling_formula():
    # Pure math: verify the per-tile eta formula matches the spec
    denoise_min, denoise_max, eta = 0.05, 0.35, 1.0
    drange = denoise_max - denoise_min

    eta_at_min = eta * (denoise_min - denoise_min) / drange
    eta_at_max = eta * (denoise_max - denoise_min) / drange
    eta_at_mid = eta * ((denoise_min + denoise_max) / 2 - denoise_min) / drange

    assert eta_at_min == pytest.approx(0.0)
    assert eta_at_max == pytest.approx(1.0)
    assert eta_at_mid == pytest.approx(0.5)


def test_adaptive_detail_uses_sample_custom_not_sample():
    comfy.sample.sample_custom.reset_mock()

    node = LLMAdaptiveTileDetailer()
    node.detail(
        model=_make_model_mock(),
        upscaled_latent={"samples": torch.zeros(1, 4, 32, 32)},
        positive=[], negative=[],
        seed=0, steps=20, cfg=7.0,
        sampler_name="euler", scheduler="normal",
        scoring_method="gradient_magnitude",
        denoise_min=0.05, denoise_max=0.35,
        curve=1.5, tile_size=256, overlap=0,
        crop_to_tiles=False,
        noise_type="gaussian", eta=1.0,
    )

    assert comfy.sample.sample_custom.call_count > 0


def test_adaptive_detail_eta_varies_across_tiles():
    # High-score tiles get more eta than low-score tiles
    captured_etas = []
    original = comfy.samplers.ksampler

    def capturing(name, extra_options=None):
        if extra_options and 'eta' in extra_options:
            captured_etas.append(extra_options['eta'])
        return original(name, extra_options=extra_options)

    comfy.samplers.ksampler = capturing
    try:
        # Use a canvas with high contrast so scores differ across tiles
        canvas = torch.zeros(1, 4, 32, 32)
        canvas[:, :, 16:, 16:] = 1.0
        node = LLMAdaptiveTileDetailer()
        node.detail(
            model=_make_model_mock(),
            upscaled_latent={"samples": canvas},
            positive=[], negative=[],
            seed=0, steps=20, cfg=7.0,
            sampler_name="euler_ancestral", scheduler="normal",
            scoring_method="gradient_magnitude",
            denoise_min=0.05, denoise_max=0.35,
            curve=1.0, tile_size=256, overlap=0,
            crop_to_tiles=False,
            noise_type="gaussian", eta=1.0,
        )
    finally:
        comfy.samplers.ksampler = original

    assert len(captured_etas) > 0
    assert max(captured_etas) > min(captured_etas), "eta should vary across tiles"


def test_adaptive_detail_input_types_include_noise_type_and_eta():
    required = LLMAdaptiveTileDetailer.INPUT_TYPES()["required"]
    assert "noise_type" in required
    assert "eta" in required
```

- [ ] **Step 2: Run new tests to confirm they fail**

```
pytest tests/test_adaptive_detailer.py::test_adaptive_eta_scaling_formula tests/test_adaptive_detailer.py::test_adaptive_detail_uses_sample_custom_not_sample tests/test_adaptive_detailer.py::test_adaptive_detail_eta_varies_across_tiles tests/test_adaptive_detailer.py::test_adaptive_detail_input_types_include_noise_type_and_eta -v
```

Expected: FAIL (`noise_type`/`eta` not yet in node inputs).

- [ ] **Step 3: Update `node_detailer_adaptive.py`**

Replace the imports block and `LLMAdaptiveTileDetailer.INPUT_TYPES` and `LLMAdaptiveTileDetailer.detail` as follows.

**Replace the imports at the top of the file:**

```python
import heapq
import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

if __package__:
    from .utils import (
        feather_blend_latent, _compute_center_grid, _compute_tile_coords,
        check_eta_support, prepare_noise_typed, build_tile_sampler,
        NOISE_GENERATOR_NAMES_SIMPLE,
    )
else:
    from utils import (
        feather_blend_latent, _compute_center_grid, _compute_tile_coords,
        check_eta_support, prepare_noise_typed, build_tile_sampler,
        NOISE_GENERATOR_NAMES_SIMPLE,
    )
```

**Replace `INPUT_TYPES` inside `LLMAdaptiveTileDetailer`:**

```python
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
            "scoring_method": (["otsu_threshold", "gradient_magnitude", "quadtree_density"],
                               {"default": "otsu_threshold"}),
            "denoise_min": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            "denoise_max": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            "curve": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.01}),
            "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
            "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            "crop_to_tiles": ("BOOLEAN", {"default": False}),
            "noise_type": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
            "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                              "tooltip": "Maximum SDE noise injection. Scales to 0 at denoise_min; full value at denoise_max. Use 'rk_beta' sampler for full RES4LYF eta control."}),
        }
    }
```

**Replace the `detail` method signature and Pass 2 block.**

Change the method signature to:

```python
def detail(self, model, upscaled_latent, positive, negative,
           seed, steps, cfg, sampler_name, scheduler,
           scoring_method, denoise_min, denoise_max, curve,
           tile_size, overlap, crop_to_tiles, noise_type, eta):
```

At the top of `detail`, after the grid/stride/print block and before `# --- Pass 1`, add:

```python
        model_sampling = model.get_model_object("model_sampling")
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)
        eta_supported = check_eta_support(sampler_name)
        if eta > 0.0 and not eta_supported:
            print(f"[LLMAdaptiveTileDetailer] Warning: '{sampler_name}' does not support "
                  f"eta; eta will be ignored. Use 'rk_beta' for full RES4LYF eta control.")
        drange = denoise_max - denoise_min
```

Replace the **entire Pass 2 loop** (lines starting with `# --- Pass 2`) with:

```python
        # --- Pass 2: sample each tile with its computed denoise and scaled eta ---
        pbar = ProgressBar(len(tile_coords))
        n_cols = cols + 1
        for tile_idx, (y1, x1, y2, x2) in enumerate(tile_coords):
            r = tile_idx // n_cols
            c = tile_idx % n_cols
            t_val, tile_denoise = td_pairs[tile_idx]
            score = scores[tile_idx]

            print(f"[LLMAdaptiveTileDetailer] tile ({r},{c}) "
                  f"{scoring_method}={score:.4f} t={t_val:.2f} denoise={tile_denoise:.3f}")

            if tile_denoise <= 0.0:
                pbar.update(1)
                continue

            tile_eta = eta * (tile_denoise - denoise_min) / drange if drange > 0 else 0.0
            tile_seed = seed + tile_idx
            tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

            if tile_denoise >= 1.0:
                tile_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
            else:
                new_steps = int(steps / tile_denoise)
                tile_sigmas = comfy.samplers.calculate_sigmas(
                    model_sampling, scheduler, new_steps)[-(steps + 1):]
            tile_sigmas = tile_sigmas.to(model.load_device)

            tile_sampler = build_tile_sampler(sampler_name, tile_eta, eta_supported)
            noise = prepare_noise_typed(tile_latent, tile_seed, noise_type, sigma_min, sigma_max)
            refined = comfy.sample.sample_custom(
                model, noise, cfg, tile_sampler, tile_sigmas,
                positive, negative, tile_latent,
            )

            feather_blend_latent(
                canvas, refined, y1, x1, overlap_l,
                has_left=(c > 0 and x1 < tile_coords[tile_idx - 1][3]),
                has_top=(r > 0 and y1 < tile_coords[tile_idx - n_cols][2]),
            )

            comfy.model_management.soft_empty_cache()
            pbar.update(1)
```

- [ ] **Step 4: Run all tests**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add node_detailer_adaptive.py tests/test_adaptive_detailer.py
git commit -m "feat: add eta scaling and noise_type to LLMAdaptiveTileDetailer"
```

---

## Self-Review

**Spec coverage:**
- ✅ Architecture: both nodes switch to `sample_custom` (Tasks 4 & 5)
- ✅ Noise type: RES4LYF dispatch with Gaussian fallback (Task 2)
- ✅ Eta: float input on both nodes (Tasks 4 & 5)
- ✅ Adaptive eta scales linearly with per-tile denoise (Task 5)
- ✅ `eta_supported` computed once before tile loop, passed to `build_tile_sampler` (Tasks 4 & 5)
- ✅ Console warning when eta > 0 and sampler doesn't support it (Tasks 4 & 5)
- ✅ Shared utilities in `utils/sampling_utils.py` (Task 2)
- ✅ `sigma_min`/`sigma_max` from model, not hardcoded (Tasks 4 & 5)
- ✅ Tests updated: mock `sample_custom` and `ksampler` instead of `sample` (Tasks 1 & 4 & 5)

**No placeholders:** all steps contain complete code.

**Type consistency:** `build_tile_sampler(sampler_name, tile_eta, eta_supported)` signature is consistent across Task 2 (definition), Task 3 (export), Tasks 4 & 5 (call sites). `prepare_noise_typed(tile_latent, tile_seed, noise_type, sigma_min, sigma_max)` is consistent across all tasks.
