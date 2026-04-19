# Optional ControlNet + Latent Seeding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ControlNet optional in both nodes and fix latent seeding so tile overlap is preserved in latent space rather than discarded.

**Architecture:** Move `controlnet`/`controlnet_strength` to optional inputs in both nodes. Replace zero-latent initialization with actual VAE encoding of `working_tensor`. Resize `outpaint_mask` to latent space and pass it to the sampler so overlap regions are preserved and only new regions are noised.

**Tech Stack:** PyTorch, ComfyUI internal APIs (`comfy.sample.sample`, guider), ComfyUI node INPUT_TYPES convention for optional inputs.

---

## File Map

| File | Change |
|------|--------|
| `node.py` | Optional inputs, latent init fix, mask resize + `noise_mask` wiring |
| `node_advanced.py` | Optional inputs, latent init fix, mask resize + `denoise_mask` wiring |
| `tests/test_mask_resize.py` | New — unit tests for mask resize logic |

---

### Task 1: Test the mask resize logic

The mask resize is pure PyTorch and fully testable without ComfyUI. Write and verify these tests before touching the nodes.

**Files:**
- Create: `tests/test_mask_resize.py`

- [ ] **Step 1: Create the test file**

```python
import torch
import torch.nn.functional as F
import pytest


def resize_mask_to_latent(outpaint_mask, latent_h, latent_w):
    """Resize outpaint_mask [1, H, W, 1] to latent space [1, latent_H, latent_W]."""
    mask = outpaint_mask[:, :, :, 0]  # [1, H, W]
    return F.interpolate(
        mask.unsqueeze(1),
        size=(latent_h, latent_w),
        mode='nearest'
    ).squeeze(1)  # [1, latent_H, latent_W]


def test_output_shape():
    mask = torch.ones(1, 1024, 1024, 1)
    result = resize_mask_to_latent(mask, 128, 128)
    assert result.shape == (1, 128, 128)


def test_all_ones_preserved():
    mask = torch.ones(1, 1024, 1024, 1)
    result = resize_mask_to_latent(mask, 128, 128)
    assert result.all()


def test_all_zeros_preserved():
    mask = torch.zeros(1, 1024, 1024, 1)
    result = resize_mask_to_latent(mask, 128, 128)
    assert not result.any()


def test_left_overlap_region():
    """Left 154px of a 1024px-wide mask should be 0, rest 1 — matches overlap_percent=0.15."""
    overlap_x = 154
    mask = torch.ones(1, 1024, 1024, 1)
    mask[:, :, :overlap_x, :] = 0.0
    result = resize_mask_to_latent(mask, 128, 128)
    latent_overlap = round(128 * (overlap_x / 1024))
    assert result[:, :, :latent_overlap].sum() == 0
    assert result[:, :, latent_overlap:].sum() > 0


def test_asymmetric_canvas():
    """Non-square canvas (tile + overlap in one axis only)."""
    mask = torch.ones(1, 1024, 1178, 1)
    mask[:, :, :154, :] = 0.0
    result = resize_mask_to_latent(mask, 128, 148)
    assert result.shape == (1, 128, 148)
```

- [ ] **Step 2: Run tests**

```bash
cd E:/StableDiffusion/ComfyUI/custom_nodes/comfyui-llm-tile
python -m pytest tests/test_mask_resize.py -v
```

Expected: all 5 tests PASS. (No ComfyUI dependency — pure torch.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_mask_resize.py
git commit -m "test: add mask resize unit tests"
```

---

### Task 2: Make ControlNet optional in `node.py`

**Files:**
- Modify: `node.py`

- [ ] **Step 1: Move controlnet inputs to optional section**

In `node.py`, find `INPUT_TYPES`. The `required` dict currently has:
```python
"controlnet": ("CONTROL_NET",),
"controlnet_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
```

Remove both from `required` and add an `optional` section after `required`:

```python
return {
    "required": {
        "json_tile_prompts": ("STRING", {"multiline": True}),
        "global_positive": ("STRING", {"multiline": True, "default": ""}),
        "global_negative": ("STRING", {"multiline": True, "default": ""}),
        "grid_width": ("INT", {"default": 4, "min": 1, "max": 16}),
        "grid_height": ("INT", {"default": 6, "min": 1, "max": 16}),
        "tile_width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
        "tile_height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
        "overlap_percent": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.5, "step": 0.01}),
        "model": ("MODEL",),
        "clip": ("CLIP",),
        "vae": ("VAE",),
        "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
        "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
        "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
        "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
        "seamlessX": ("BOOLEAN", {"default": True, "tooltip": "If true, side of image will be seamless. (2+ tiles)"}),
        "seamlessY": ("BOOLEAN", {"default": False, "tooltip": "If true, top/bottom of image will be seamless. (2+ tiles)"}),
    },
    "optional": {
        "controlnet": ("CONTROL_NET",),
        "controlnet_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
    }
}
```

- [ ] **Step 2: Update function signature to accept optional controlnet**

Find the `generate_tiled_image` method signature:

```python
def generate_tiled_image(self, json_tile_prompts, global_positive, global_negative,
                         grid_width, grid_height, tile_width, tile_height,
                         overlap_percent, controlnet, controlnet_strength,
                         seed, model, clip, vae, sampler_name, scheduler, steps, cfg, seamlessX, seamlessY):
```

Replace with (move `controlnet` and `controlnet_strength` to keyword args with defaults):

```python
def generate_tiled_image(self, json_tile_prompts, global_positive, global_negative,
                         grid_width, grid_height, tile_width, tile_height,
                         overlap_percent, seed, model, clip, vae,
                         sampler_name, scheduler, steps, cfg, seamlessX, seamlessY,
                         controlnet=None, controlnet_strength=0.7):
```

- [ ] **Step 3: Commit**

```bash
git add node.py
git commit -m "feat: make controlnet optional in standard node"
```

---

### Task 3: Fix latent seeding and wire mask in `node.py`

**Files:**
- Modify: `node.py`

- [ ] **Step 1: Replace zero-latent with encoded working_tensor**

Find this block (around line 253):
```python
# Get latent shape for the variable canvas
with torch.no_grad():
    latent_shape = vae.encode(working_tensor).shape

# Create empty latent tensor with actual generation shape
latent_image = torch.zeros(latent_shape, device=device)
```

Replace with:
```python
with torch.no_grad():
    latent_image = vae.encode(working_tensor).to(device)
```

- [ ] **Step 2: Resize outpaint_mask to latent space**

Immediately after the latent encoding above, add:

```python
mask = outpaint_mask[:, :, :, 0]  # [1, H, W]
latent_mask = torch.nn.functional.interpolate(
    mask.unsqueeze(1),
    size=(latent_image.shape[2], latent_image.shape[3]),
    mode='nearest'
).squeeze(1).to(device)  # [1, latent_H, latent_W]
```

- [ ] **Step 3: Pass noise_mask to comfy.sample.sample**

Find the sample call (around line 276):
```python
samples = comfy.sample.sample(
    model,
    noise,
    steps,
    cfg,
    sampler_name,
    scheduler,
    conditioning[0],
    conditioning[1],
    latent_image
)
```

Replace with:
```python
samples = comfy.sample.sample(
    model,
    noise,
    steps,
    cfg,
    sampler_name,
    scheduler,
    conditioning[0],
    conditioning[1],
    latent_image,
    noise_mask=latent_mask
)
```

- [ ] **Step 4: Verify tests still pass**

```bash
python -m pytest tests/test_mask_resize.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add node.py
git commit -m "feat: fix latent seeding and wire outpaint_mask in standard node"
```

---

### Task 4: Make ControlNet optional in `node_advanced.py`

**Files:**
- Modify: `node_advanced.py`

- [ ] **Step 1: Move controlnet inputs to optional section**

In `node_advanced.py`, find `INPUT_TYPES`. The `required` dict currently has:
```python
"controlnet": ("CONTROL_NET",),
"controlnet_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
```

Remove both from `required` and add an `optional` section:

```python
return {
    "required": {
        "json_tile_prompts": ("STRING", {"multiline": True}),
        "grid_width": ("INT", {"default": 4, "min": 1, "max": 16}),
        "grid_height": ("INT", {"default": 6, "min": 1, "max": 16}),
        "tile_width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
        "tile_height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
        "overlap_percent": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.5, "step": 0.01}),
        "noise": ("NOISE",),
        "guider": ("GUIDER",),
        "sampler": ("SAMPLER",),
        "sigmas": ("SIGMAS",),
        "clip": ("CLIP",),
        "vae": ("VAE",),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
        "seamlessX": ("BOOLEAN", {"default": True, "tooltip": "If true, left/right of image will be seamless. (2+ tiles)"}),
        "seamlessY": ("BOOLEAN", {"default": False, "tooltip": "If true, top/bottom of image will be seamless. (2+ tiles)"}),
    },
    "optional": {
        "controlnet": ("CONTROL_NET",),
        "controlnet_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
    }
}
```

- [ ] **Step 2: Update function signature**

Find:
```python
def generate_tiled_image(self, json_tile_prompts, grid_width, grid_height,
                         tile_width, tile_height, overlap_percent,
                         controlnet, controlnet_strength, seed, noise, guider,
                         sampler, sigmas, clip, vae, seamlessX, seamlessY):
```

Replace with:
```python
def generate_tiled_image(self, json_tile_prompts, grid_width, grid_height,
                         tile_width, tile_height, overlap_percent,
                         noise, guider, sampler, sigmas, clip, vae,
                         seed, seamlessX, seamlessY,
                         controlnet=None, controlnet_strength=0.7):
```

- [ ] **Step 3: Commit**

```bash
git add node_advanced.py
git commit -m "feat: make controlnet optional in advanced node"
```

---

### Task 5: Fix latent seeding and wire mask in `node_advanced.py`

**Files:**
- Modify: `node_advanced.py`

- [ ] **Step 1: Replace zero-latent with encoded working_tensor**

Find (around line 241):
```python
# Get latent shape for the variable canvas
with torch.no_grad():
    latent_shape = vae.encode(working_tensor).shape

# Create empty latent tensor with variable shape
latent_image = torch.zeros(latent_shape, device=device)
```

Replace with:
```python
with torch.no_grad():
    latent_image = vae.encode(working_tensor).to(device)
```

- [ ] **Step 2: Resize outpaint_mask to latent space**

Immediately after the latent encoding, add:

```python
mask = outpaint_mask[:, :, :, 0]  # [1, H, W]
latent_mask = torch.nn.functional.interpolate(
    mask.unsqueeze(1),
    size=(latent_image.shape[2], latent_image.shape[3]),
    mode='nearest'
).squeeze(1).to(device)  # [1, latent_H, latent_W]
```

- [ ] **Step 3: Pass denoise_mask to guider.sample**

Find (around line 280):
```python
samples = guider.sample(
    tile_noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    disable_pbar=False,
    seed=current_seed
)
```

Replace with:
```python
samples = guider.sample(
    tile_noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=latent_mask,
    disable_pbar=False,
    seed=current_seed
)
```

- [ ] **Step 4: Verify tests still pass**

```bash
python -m pytest tests/test_mask_resize.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add node_advanced.py
git commit -m "feat: fix latent seeding and wire outpaint_mask in advanced node"
```

---

### Task 6: Manual verification in ComfyUI

ComfyUI nodes cannot be exercised outside ComfyUI. Verify the following scenarios by running workflows.

- [ ] **Scenario A: Standard node without ControlNet**
  - Connect the standard Tiled Image Generator with no ControlNet input
  - Run a 2×2 grid
  - Expected: generates successfully, tile seams visible but no crash

- [ ] **Scenario B: Standard node with ControlNet**
  - Connect a ControlNet Union SDXL
  - Run a 2×2 grid
  - Expected: same output quality as before this change

- [ ] **Scenario C: Advanced node without ControlNet (Z-Turbo)**
  - Connect the Advanced node with a Z-Turbo guider/sampler/sigmas, no ControlNet
  - Run a 2×2 grid
  - Expected: generates successfully with overlap regions carrying over from neighbor tiles

- [ ] **Scenario D: Advanced node with ControlNet**
  - Connect ControlNet to the Advanced node
  - Run a 2×2 grid
  - Expected: ControlNet applies, same quality as before

- [ ] **Step: Final commit if any fixes were needed**

```bash
git add -p
git commit -m "fix: address issues found during manual ComfyUI verification"
```
