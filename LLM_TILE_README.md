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

**Tiled Image Generator** (`node.py`) — standard KSampler, uses ControlNet Union SDXL for outpainting coherence.

**Tiled Image Generator Advanced** (`node_advanced.py`) — custom sampler/guider inputs, same generation logic, more flexibility for Flux and other pipelines.

Both nodes take a `json_tile_prompts` string: a JSON array where each object has a `position` (`x`, `y`) and a `prompt`. The grid processes tiles left-to-right, top-to-bottom. Each tile gets its own prompt. Each tile after the first copies the overlapping edge from its neighbor and generates into that seed.

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

The `global_positive` and `global_negative` inputs (standard node) are appended to every tile prompt at generation time. Use them for style anchors and universal suppressions — anything that should hold across the whole image.

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `grid_width` | 4 | Columns of tiles |
| `grid_height` | 6 | Rows of tiles |
| `tile_width` | 1024 | Width of each tile in pixels |
| `tile_height` | 1024 | Height of each tile in pixels |
| `overlap_percent` | 0.15 | 15–25% recommended |
| `controlnet_strength` | 0.7 | Higher = stronger seam coherence |
| `seamlessX` | true | Wraps last column into first for seamless repeat |
| `seamlessY` | false | Wraps last row into first |

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rickyars/comfyui-llm-tile.git
```

Restart ComfyUI.

---

## Credit

Technique by [Roope Rainisto](https://x.com/rainisto/status/1891520314493870458). Implementation by rickyars.
