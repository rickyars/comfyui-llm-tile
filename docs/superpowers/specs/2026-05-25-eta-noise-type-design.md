# Eta Control & Noise Types for Tile Detailers

**Date:** 2026-05-25  
**Scope:** `node_detailer.py`, `node_detailer_adaptive.py`, new `utils/sampling_utils.py`

## Problem

Both detailer nodes call `comfy.sample.sample()`, which creates a sampler internally with empty `extra_options`. There is no way to inject eta (SDE noise injection amount) or vary the initial noise distribution through this API.

## Goals

1. Add per-tile **eta** control to both detailer nodes.
2. In the adaptive detailer, eta scales linearly with per-tile denoise so low-denoise tiles run ODE (deterministic) and high-denoise tiles run full SDE.
3. Add **noise type** selection for the initial tile noise, backed by RES4LYF's noise generators when available.

## Architecture Change: `sample()` → `sample_custom()`

Both nodes switch from `comfy.sample.sample()` to `comfy.sample.sample_custom()`. This requires building the sampler object and sigmas manually per tile.

**Sigma slicing** (replicates `KSampler.set_steps` exactly):
```python
model_sampling = model.get_model_object("model_sampling")
if tile_denoise >= 1.0:
    tile_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
else:
    new_steps = int(steps / tile_denoise)
    tile_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, new_steps)[-(steps + 1):]
tile_sigmas = tile_sigmas.to(model.load_device)
```

**Sampler construction with eta:**
```python
extra_opts = {'eta': tile_eta} if eta_supported else {}
sampler = comfy.samplers.ksampler(sampler_name, extra_options=extra_opts)
```

`eta_supported` is determined once before the tile loop by inspecting the sampler function's signature:
```python
import inspect
fn = getattr(comfy.samplers.k_diffusion_sampling, f"sample_{sampler_name}", None)
eta_supported = fn is not None and 'eta' in inspect.signature(fn).parameters
```

If eta > 0 and `eta_supported` is False, a one-time console warning is printed. The sampler still runs without eta.

**Sampler compatibility notes:**
- Standard SDE/ancestral samplers (`euler_ancestral`, `dpm_2_ancestral`, etc.): eta supported natively.
- `rk_beta` (RES4LYF unified sampler): eta supported via explicit kwarg. Use `rk_beta` with `rk_type` in `extra_options` for full RES4LYF eta control.
- `res_2m`, `res_2s`, etc. (thin wrappers): do NOT accept eta. Use `rk_beta` instead.
- ODE samplers (`euler`, `dpm++_2m`, etc.): eta not applicable; correctly skipped.

## Feature 1: Noise Types

### Import strategy

At module level, try to import from sibling custom node. Fall back gracefully:

```python
try:
    from RES4LYF.beta.noise_classes import (
        NOISE_GENERATOR_CLASSES_SIMPLE,
        NOISE_GENERATOR_NAMES_SIMPLE,
    )
    _RES4LYF_NOISE = True
except ImportError:
    NOISE_GENERATOR_NAMES_SIMPLE = ("gaussian",)
    _RES4LYF_NOISE = False
```

### New input

Both nodes get a `noise_type` dropdown using `NOISE_GENERATOR_NAMES_SIMPLE` as the options list. Default: `"gaussian"`.

### Noise generation per tile

Replaces `comfy.sample.prepare_noise(tile_latent, tile_seed, None)`:

```python
def prepare_noise_typed(tile_latent, seed, noise_type, sigma_min, sigma_max):
    if not _RES4LYF_NOISE or noise_type == "gaussian":
        return comfy.sample.prepare_noise(tile_latent, seed, None)
    cls = NOISE_GENERATOR_CLASSES_SIMPLE[noise_type]
    gen = cls(x=tile_latent, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    return gen(sigma=sigma_max, sigma_next=sigma_min)
```

`sigma_min` and `sigma_max` are read from `model.get_model_object("model_sampling")` — real model values, not hardcoded SDXL constants.

## Feature 2: Eta Control

### Regular detailer (`LLMTileSequentialDetailer`)

New input: `eta` FLOAT, default=0.0, min=0.0, max=2.0, step=0.01.

Applied uniformly to every tile. `tile_eta = eta`.

### Adaptive detailer (`LLMAdaptiveTileDetailer`)

New input: `eta` FLOAT, default=1.0, min=0.0, max=2.0, step=0.01. Tooltip: "Maximum eta applied to highest-denoise tiles. Scales to 0 at denoise_min."

Per-tile eta scales linearly with denoise:
```python
drange = denoise_max - denoise_min
tile_eta = eta * (tile_denoise - denoise_min) / drange if drange > 0 else 0.0
```

`denoise_min` and `denoise_max` are the existing node inputs, already in scope during the tile loop.

## Shared Utilities (`utils/sampling_utils.py`)

Two functions extracted here, used by both nodes:

```python
def prepare_noise_typed(tile_latent, seed, noise_type, sigma_min, sigma_max) -> Tensor
def build_tile_sampler(sampler_name, tile_eta, eta_supported) -> KSAMPLER
```

`eta_supported` is computed once per node call (before the tile loop) by the node itself and passed into `build_tile_sampler` on each iteration — it is not recomputed per tile.

`calculate_tile_sigmas` is kept inline in each node (3 lines, clearer with `model` in scope).

## New Inputs Summary

| Node | New input | Type | Default |
|------|-----------|------|---------|
| Both | `noise_type` | dropdown | `"gaussian"` |
| Both | `eta` | FLOAT | 0.0 (regular) / 1.0 (adaptive) |

## Testing

Existing tests in `tests/test_detailer.py` and `tests/test_adaptive_detailer.py` mock `comfy.sample.sample` — these mocks must be updated to mock `comfy.sample.sample_custom` and `comfy.samplers.ksampler` instead. New test cases:

- `noise_type="gaussian"` with RES4LYF absent falls back correctly.
- `noise_type="brownian"` with RES4LYF present calls the right generator class.
- `eta=0.0` produces no eta in extra_options (or eta_supported=False path).
- `eta=1.0` on a sampler that supports it passes eta correctly.
- Adaptive detailer: tile at `denoise_min` gets `tile_eta=0`; tile at `denoise_max` gets `tile_eta=eta`.
- Sigma slicing at various denoise values matches KSampler's output.
