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

**Tiled Image Generator** (`node.py`) — standard KSampler with model, CLIP, VAE, sampler name, scheduler, steps, and CFG inputs. Uses ControlNet for outpainting coherence at tile seams.

**Tiled Image Generator Advanced** (`node_advanced.py`) — custom sampler, guider, noise, and sigmas inputs. Designed for Flux and other pipelines that require non-standard sampling. Same generation logic, more flexibility.

Both nodes take a `json_tile_prompts` string: a JSON array where each object has a `position` (`x`, `y`) and a `prompt`. The grid processes tiles left-to-right, top-to-bottom. Each tile gets its own prompt. Each tile after the first copies the overlapping edge from its neighbor and generates into that seed region.

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
| `controlnet_strength` | 0.7 | Higher = stronger seam coherence |
| `seed` | 0 | Base seed; each tile increments by 1 |
| `seamlessX` | true | Wraps last column into first for seamless horizontal repeat |
| `seamlessY` | false | Wraps last row into first for seamless vertical repeat |

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
