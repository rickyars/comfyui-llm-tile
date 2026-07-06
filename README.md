# comfyui-llm-tile

A ComfyUI custom node for LLM-directed tiled image generation.

---

## The idea

Roope Rainisto made a piece called *The Swimming Hall* for Christie's "Augmented Intelligence" auction in February 2025. 11,519 x 7,936 pixels. The finished prompts, taken together, were almost one A4 page long.

He did not write one big prompt. He wrote 24 small ones.

The method: give an LLM a master concept and ask it to imagine the finished image, then describe what it would see in each tile. The LLM outputs one self-contained prompt per tile. Those feed into the image generator. The result is stitched together with overlapping seeds for coherence.

This is the Brenizer Method applied to AI image generation. In photography, Brenizer shoots a panorama with a fast portrait lens — multiple frames, tight focus, stitched together — to get a wide field of view with shallow depth of field impossible in a single shot. Here, the constraint being broken is prompt locality: a single prompt targets the entire image. Tile-specific prompts let each region of the composition get its own generative pass, directed by a model that has already imagined the whole.

The diagram Rainisto kept showing: AI splits into genAI (image generation) and LLM (language model). The prompt sits between them. Conventionally, the creative work happens in that purple wiggly line — the translation from prompt to image. His move was to push the creativity upstream, into the LLM layer, before the image generator ever runs.

---

## Why it works

A 24-tile grid generates far more compositional complexity than a single prompt can. Each tile prompt can specify local content, local lighting, local subject position. The genAI model treats each tile as its own problem. The overlapping seed regions stitch adjacent tiles without hard seams.

The LLM does not hallucinate global coherence and hope for the best. It describes local reality. The image generator fills it in. The seam-handling code makes the joins invisible.

---

## How it works

Two nodes:

**Tiled Image Generator** (`node.py`) — standard KSampler with model, CLIP, VAE, sampler name, scheduler, steps, and CFG inputs. Drives seam coherence with a ControlNet or a model patch (see *Model coherence* below).

**Tiled Image Generator Advanced** (`node_advanced.py`) — custom sampler, guider, noise, and sigmas inputs. Designed for Flux and other pipelines that require non-standard sampling. Same generation logic, more flexibility.

Both nodes take a `json_tile_prompts` string: a JSON array where each object has a `position` (`x`, `y`) and a `prompt`. The grid processes tiles left-to-right, top-to-bottom. Each tile gets its own prompt. Each tile after the first copies the overlapping edge from its neighbor and generates into that seed region.

---

## Model coherence

Each tile after the first is generated with its neighbor's overlapping pixels as context. How those pixels steer generation depends on the model family — wire the matching coherence input:

**SDXL / Flux — `controlnet` input.** Load a tile/inpaint ControlNet and connect it to the `controlnet` input. The neighbor pixels become the control image.

> **SDXL Union ControlNet** (e.g. `controlnet++_union_sdxl`) does nothing until you set its control type. Insert `ControlNetLoader → SetUnionControlNetType` (type `tile` or `repaint`) → `controlnet` input. Without this you get a checkerboard of independent tiles.

**z-image (Turbo) / Qwen — `model_patch` input.** These use a DiffSynth/Fun inpaint model patch instead of a conditioning ControlNet. Load `Z-Image-Turbo-Fun-Controlnet-Union` (2.1 for inpaint) via `ModelPatchLoader` and connect it to the `model_patch` input. The node feeds the neighbor pixels in as the inpaint image plus a keep-mask of the overlap zone; the kept pixels are composited back after decode so seams stay exact.

If neither input is wired the node falls back to independent per-tile generation (no seam coherence).

The generator resizes empty latents to each model's channel count automatically, so 16-channel and video-format latents (Flux, Krea2/Wan21, z-image) work without extra wiring.

---

## JSON format

```json
[
  {
    "position": { "x": 1, "y": 1 },
    "prompt": "aerial view of an Olympic swimming pool, upper left corner, blue lane lines, crowded with swimmers, midday light"
  },
  {
    "position": { "x": 2, "y": 1 },
    "prompt": "aerial view of an Olympic swimming pool, upper center, diving platform visible at left edge, spectators packed on bleachers, midday light"
  }
]
```

Position is 1-indexed. `x` is column, `y` is row. The array must contain exactly `grid_width * grid_height` objects.

`global_positive` and `global_negative` (standard node) are appended to every tile prompt at generation time. Use them for style anchors and universal suppressions — anything that should hold across the whole image. Do not repeat style tokens in the tile prompts themselves.

---

## Parameters

### Both nodes

| Parameter | Default | Notes |
|---|---|---|
| `grid_width` | 4 | Columns of tiles |
| `grid_height` | 6 | Rows of tiles |
| `tile_width` | 1024 | Width of each tile in pixels |
| `tile_height` | 1024 | Height of each tile in pixels |
| `overlap_percent` | 0.15 | 15–25% recommended |
| `control_strength` | 0.7 | Strength of the coherence signal (ControlNet or model patch). Higher = stronger seam coherence |
| `seed` | 0 | Base seed; each tile increments by 1 |
| `seamlessX` | true | Wraps last column into first for seamless horizontal repeat |
| `seamlessY` | false | Wraps last row into first for seamless vertical repeat |
| `controlnet` | — | Optional. SDXL/Flux coherence ControlNet (see *Model coherence*) |
| `model_patch` | — | Optional. z-image/Qwen DiffSynth inpaint model patch (see *Model coherence*) |

### Standard node only

| Parameter | Default | Notes |
|---|---|---|
| `global_positive` | — | Style tokens appended to every tile prompt |
| `global_negative` | — | Negative prompt applied to every tile |
| `steps` | 20 | Sampler steps |
| `cfg` | 7.0 | Classifier-free guidance scale |
| `sampler_name` | — | KSampler sampler selection |
| `scheduler` | — | KSampler scheduler selection |

---

## Generating tile prompts with an LLM

Use this template with Claude (or any capable LLM). Fill in the bracketed fields and paste the output directly into the `json_tile_prompts` input in ComfyUI.

```
I am generating a large tiled image using a ComfyUI node that processes each tile independently.

The image is a [GRID_WIDTH] x [GRID_HEIGHT] grid of [TILE_WIDTH]x[TILE_HEIGHT]px tiles.
Total output: [TOTAL_WIDTH] x [TOTAL_HEIGHT] pixels.

Master concept: [DESCRIBE THE SCENE — subject, mood, lighting, vantage point, time of day, visual register]

Global style (applied to all tiles, do not repeat in tile prompts): [STYLE TOKENS — e.g. "aerial photography, hyperrealistic, soft midday light, Hasselblad"]

Imagine the finished artwork completely in your mind. Then describe what you would see in each tile.

Rules:
- Each tile prompt must be self-contained. Do not reference other tiles.
- Describe only what is visible within that tile's region of the frame.
- Be specific about local content: what object, what action, what light condition exists in this exact region.
- Maintain consistent vantage point, lighting direction, and atmosphere across all tiles.
- Do not repeat the global style tokens — those are handled separately.
- Tile position (x, y) is 1-indexed. x = column (left to right), y = row (top to bottom).

Return a JSON array with exactly [TOTAL_TILES] objects. No explanation, no preamble, no markdown fences. Raw JSON only.

Format:
[
  {
    "position": { "x": 1, "y": 1 },
    "prompt": "..."
  },
  ...
]
```

**Prompt length per tile.** 30–80 words per tile. Enough to specify local content precisely, not so much that you're describing the whole image again.

**Global positive vs. tile prompts.** Style tokens, medium, lighting register — put these in `global_positive` in ComfyUI. The node appends it to every tile at generation time. Repeating those tokens in each tile prompt doubles them and wastes context.

**Concept types that work well:**
- Scenes with natural regional variation (crowds, cityscapes, landscapes, interiors)
- Compositions with a clear vantage point that holds across the whole image (aerial, straight-on elevation, isometric)
- Subject matter where local detail matters — tile-level specificity is where the complexity comes from

**Concept types that need extra care:**
- Single subjects that span the whole frame — the LLM needs to think about which body part or detail occupies each tile
- Strong geometric compositions — describe the geometry explicitly per tile or it will drift

**SeamlessX.** If `seamlessX` is enabled, tell the LLM in the master concept: *"the composition should repeat seamlessly left to right — the right edge of the last column should match the left edge of the first."*

---

## Tiled detailers

Two nodes for refining upscaled images tile by tile. Wire an upscaled latent into either node in place of a KSampler.

Both detailers handle 16-channel and video-format latents, so SDXL, Flux, z-image, and Krea2/Wan21 all work. Single-frame video latents (Krea2) are supported; the node squeezes and restores the temporal axis around sampling.

---

### Tiled Image Detailer

Applies a single denoise value to every tile. Good starting point; use when the image content is fairly uniform or you want predictable, consistent refinement across the whole canvas.

| Parameter | Default | Notes |
|---|---|---|
| `denoise` | 0.25 | Applied uniformly to every tile |
| `tile_size` | 1024 | Tile size in pixels |
| `overlap` | 64 | Overlap between adjacent tiles in pixels. Tiles are feather-blended in overlap zones using a smoothstep curve to hide seams. |
| `edge_mode` | `center` | How the grid edges are handled when the image size is not a multiple of `tile_size`. `center`: diffuse the centered grid and leave the outer strips as the original upscale (output keeps original size). `crop`: crop the output down to the detailed region. `pad`: keep the same tile grid as `center` and add one edge tile per uncovered strip — the latent is edge-replicated outward so those edge tiles have data, everything is diffused, then the output is cropped back to original size. Adds up to one extra tile per padded axis. |
| `noise_type` | `gaussian` | Initial noise distribution per tile. `gaussian` is the standard ComfyUI default. Other types (brownian, uniform, etc.) require [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF). |
| `eta` | 1.0 | SDE noise injection per step. 0 = deterministic (ODE path). 1 = standard ancestral. Only applies to ancestral/SDE samplers: `euler_ancestral`, `dpmpp_sde`, `dpmpp_2s_ancestral`, `dpmpp_2m_sde`, `dpmpp_3m_sde`, `rk_beta`. ODE samplers (`euler`, `dpm++_2m`, etc.) ignore this. Values above 1.0 inject more noise than the SDE derivation calls for — adds texture but risks incoherence at low denoise. |

The tile grid uses whole user-sized tiles centered on the image. If the image size is not an exact multiple of `tile_size`, the outside strips are not covered by the centered grid; use `edge_mode` to control them — `center` leaves them untouched, `crop` removes them, and `pad` keeps the same grid and adds an edge tile per strip (edge-replicating the latent so they have data) so they get detailed too, then crops back to the original size.

---

### Adaptive Tiled Image Detailer

Measures each tile before sampling it, then scales the denoise strength accordingly. Three scoring methods are available: `otsu_threshold`, `quadtree_density`, and `blur_sensitivity`.

Scores are **absolute**: a tile's denoise depends only on its own content, never on the other tiles in the image. This makes the node safe for batch processing with a single `denoise_min`/`denoise_max`: a uniformly soft image sits near `denoise_min` everywhere instead of having its least-soft tile promoted to `denoise_max`. (Earlier versions min-max normalized scores within each image, which made every image stretch to `denoise_max` somewhere; the `gradient_magnitude` method only made sense under that normalization and was removed with it.)

Outputs three things: the refined latent, a denoise map, and a scoring map. The denoise map is a tile heatmap using the viridis colormap (dark purple = low denoise, teal = mid, yellow = high denoise). The scoring map is a debug visualization specific to the active scoring method.

#### How it works

**Pass 1** computes the centered whole-tile grid once. The same tile coordinates are used for scoring, the denoise heatmap, and sampling.

Tile scores land directly on an absolute [0, 1] scale (0 = no detail, 1 = maximal detail), are mapped through the `curve` exponent, and scaled to [denoise_min, denoise_max].

After scoring, each tile's raw score is blended with the average of its 4-connected neighbors (70% own, 30% neighbor average). This prevents abrupt denoise jumps between adjacent tiles while preserving the coarse adaptive distribution.

**Pass 2** runs the standard sampling loop with each tile's computed denoise value.

#### Scoring methods

**`otsu_threshold`** — Averages latent channels into a single intensity map, normalizes to [0, 1], computes a global Otsu threshold, and scores each tile by its fraction of bright-class (above-threshold) pixels — an absolute coverage fraction. The scoring map is a black/white image where white = above threshold. Works well when the image has a clear bimodal intensity distribution. Note it measures *subject coverage*, not sharpness: a soft image with a bright subject still scores high.

**`quadtree_density`** — Runs a threshold-based quadtree over the entire canvas: a cell subdivides only while its detail (mean per-channel latent std) exceeds `split_threshold`, down to a 4-latent-pixel floor. Each tile is scored by its leaf density, normalized so 1.0 = subdivided to the floor everywhere and 0.0 = no detail anywhere. A soft image genuinely produces large leaves everywhere and low scores in every tile — the recommended method for batch processing. The scoring map shows the quadtree cell outlines (white on black) — large cells = flat, small cells = complex.

**`blur_sensitivity`** — Scores each tile by how much of its gradient energy a small 3×3 blur destroys: `1 - grad_energy(blurred) / grad_energy(original)`. Fine detail is annihilated by blurring (score near 1); smooth or already-soft content barely changes (score near 0). The ratio cancels the latent's units, so no threshold or calibration is needed at all — the most direct measure of actual sharpness. The scoring map is a grayscale image where white = fine detail a blur would destroy.

#### Parameters

| Parameter | Default | Notes |
|---|---|---|
| `scoring_method` | `otsu_threshold` | `otsu_threshold`, `quadtree_density`, or `blur_sensitivity` |
| `denoise_min` | 0.05 | Denoise applied to a zero-score tile. Near zero freezes it. |
| `denoise_max` | 0.35 | Denoise applied to a full-score (1.0) tile. Soft images may never reach it — that's the point. |
| `split_threshold` | 0.35 | Optional, `quadtree_density` only. A cell subdivides while its mean per-channel latent std exceeds this. Lower = more of the image counts as detailed. VAE latents are roughly unit-variance, so the default is a reasonable starting point across models. |
| `curve` | 1.5 | Controls how denoise is distributed across tiles. See below. |
| `tile_size` | 1024 | Tile size in pixels |
| `overlap` | 64 | Overlap between adjacent tiles in pixels. Tiles are feather-blended using a smoothstep curve to hide seams. |
| `edge_mode` | `center` | How the grid edges are handled when the image size is not a multiple of `tile_size`. `center`: diffuse the centered grid and leave the outer strips as the original upscale. `crop`: crop the output latent and debug images down to the detailed region. `pad`: keep the same tile grid as `center` and add one edge tile per uncovered strip (the latent is edge-replicated outward so they have data), diffuse everything, then crop back to original size (debug maps are cropped to match). Adds up to one extra tile per padded axis. |
| `noise_type` | `gaussian` | Initial noise distribution per tile. `gaussian` is standard. Other types require [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF). `brownian` is a good first alternative for portraits and fabric. |
| `eta_min` | 0.0 | Eta applied to the lowest-denoise tiles. 0 = deterministic ODE for those tiles. Only applies to ancestral/SDE samplers (see Tiled Image Detailer note above). |
| `eta_max` | 1.0 | Eta applied to the highest-denoise tiles. Scales linearly from `eta_min` at `denoise_min` to `eta_max` at `denoise_max`. Values above 1.0 amplify noise beyond the SDE derivation — useful for texture but risky above 1.3. |

#### How `curve` works

The node raises each tile's absolute [0, 1] score to the power of `curve` before mapping to the denoise range:

```
t_curved = t ^ curve
denoise  = denoise_min + t_curved * (denoise_max - denoise_min)
```

`t ^ curve` is still 0 at the low end and 1 at the high end. The endpoints are fixed; what changes is the shape of the distribution in between.

**`curve = 1.0` (linear):** denoise scales evenly with the normalized tile score. A tile at 50% gets 50% of the way between denoise_min and denoise_max.

**`curve > 1.0` (e.g. 1.5, 2.0):** mid and low scores are pushed toward denoise_min. Only the highest-scoring tiles approach denoise_max.

**`curve < 1.0` (e.g. 0.5):** mid scores are boosted toward denoise_max. More of the image gets significant refinement while the lowest-scoring tiles remain near denoise_min.

Think of `curve` as a contrast control for the denoise distribution, not a global strength control. `denoise_min` and `denoise_max` set the floor and ceiling; `curve` sets how quickly the population of tiles rises from the floor.

The current `curve` is a gamma curve, not an S-curve. It preserves fixed endpoints: normalized score 0 always maps to `denoise_min`, and normalized score 1 always maps to `denoise_max`. Use `curve < 1.0` if you want most non-purple tiles to receive higher denoise. A future S-curve or threshold/knee control would be better for "purple gets almost nothing, everything else gets a lot."

#### Edge cases

If all tiles have identical scores and the score is near zero, every tile gets `denoise_min`. If all tiles have the same non-zero score, every tile gets `denoise_max`.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rickyars/comfyui-llm-tile.git
```

Restart ComfyUI.

---

## License

MIT License

---

## Credit

Technique by [Roope Rainisto](https://x.com/rainisto/status/1891520314493870458). Implementation by rickyars.
