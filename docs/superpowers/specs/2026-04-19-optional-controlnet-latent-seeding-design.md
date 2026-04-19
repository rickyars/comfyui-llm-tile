# Optional ControlNet + Latent Seeding Design

**Date**: 2026-04-19  
**Goal**: Make ControlNet optional and fix latent seeding so the node works with Z-Turbo (and any model that doesn't support ControlNet).

---

## Problem

Two bugs in the current implementation:

1. `controlnet` is a **required** input in both nodes — the node cannot run without one.
2. The overlap seeding is **half-implemented**: overlap pixels are correctly copied into `working_tensor` and `outpaint_mask` is correctly built, but then `working_tensor` is encoded only to extract its shape, and a zero-latent is used for sampling. `outpaint_mask` is never passed to the sampler. ControlNet is therefore doing 100% of the coherence work.

---

## Solution

Three surgical changes to `node.py` and `node_advanced.py`. No new files. No changes to `controlnet_utils.py`, `image_utils.py`, or any utils.

### 1. Make ControlNet optional

Move `controlnet` and `controlnet_strength` from `required` to `optional` in `INPUT_TYPES` in both nodes.

`apply_controlnet_to_conditioning` already guards `if control_net is None: return (positive, negative)` — no change needed there.

### 2. Fix latent seeding

Replace the current pattern:
```python
with torch.no_grad():
    latent_shape = vae.encode(working_tensor).shape
latent_image = torch.zeros(latent_shape, device=device)
```

With:
```python
with torch.no_grad():
    latent_image = vae.encode(working_tensor).to(device)
```

The overlap regions of `working_tensor` contain real pixel data from neighbors. The new region contains zeros — but since the mask marks those as `1` (generate), the sampler applies full noise there and the encoded value is irrelevant.

### 3. Wire `outpaint_mask` to the sampler

Resize `outpaint_mask` from pixel space to latent space and pass it to the sampler:

```python
# outpaint_mask: [1, H, W, 1] → [1, latent_H, latent_W]
mask = outpaint_mask[:, :, :, 0]  # drop channel dim → [1, H, W]
latent_mask = torch.nn.functional.interpolate(
    mask.unsqueeze(1),  # [1, 1, H, W]
    size=(latent_image.shape[2], latent_image.shape[3]),
    mode='nearest'
).squeeze(1)  # [1, latent_H, latent_W]
```

- **`node.py`**: pass as `noise_mask=latent_mask` to `comfy.sample.sample`
- **`node_advanced.py`**: pass as `denoise_mask=latent_mask` to `guider.sample` (currently hardcoded `None`)

---

## Data Flow (per tile, post-fix)

1. Copy overlap pixels into `working_tensor`, set `outpaint_mask = 0` for those regions *(unchanged)*
2. Encode `working_tensor` → real latents *(fixed)*
3. Resize mask to latent space *(new)*
4. If ControlNet connected: apply to conditioning *(unchanged, now optional)*
5. Pass real latents + mask to sampler → overlap latents preserved, new region fully noised *(fixed)*
6. Decode, extract new region, place on canvas *(unchanged)*

---

## Edge Cases

| Case | Behavior |
|------|----------|
| First tile (top-left) | Mask is all ones, full noise — identical to current behavior |
| Top row tiles (y=0, x>0) | Left overlap has real data (mask=0), top region is zeros but mask=1 → fully noised, irrelevant |
| ControlNet absent | `apply_controlnet_to_conditioning` returns conditioning unchanged |
| Seamless wrap tiles | Wrap overlap already sets mask=0 correctly, flows through unchanged |
| Device mismatch | `.to(device)` after `vae.encode` ensures latent is on correct device |

---

## Files Changed

| File | Change |
|------|--------|
| `node.py` | Optional inputs, latent init fix, mask resize + pass to `comfy.sample.sample` |
| `node_advanced.py` | Optional inputs, latent init fix, mask resize + pass to `guider.sample` |

---

## What Does Not Change

- `controlnet_utils.py` — already handles `None` correctly
- `image_utils.py` — not involved
- Tile extraction logic
- Seamless wrap logic
- Canvas placement logic
- All sampler/guider wiring in `node_advanced.py`
