# LLM Tile Prompt Template

Use this prompt to generate the JSON tile array for the `comfyui-llm-tile` node.

Paste the template into Claude, fill in the bracketed fields, and paste the output directly into the `json_tile_prompts` input in ComfyUI.

---

## Template

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

---

## Filled example — 4x6 aerial swimming pool

```
I am generating a large tiled image using a ComfyUI node that processes each tile independently.

The image is a 4 x 6 grid of 1024x1024px tiles.
Total output: 4096 x 6144 pixels.

Master concept: an Olympic outdoor swimming pool shot from directly above, packed with swimmers during a summer competition. Crowded bleachers surround the pool. Umbrellas, lane markers, diving platforms, officials at the edges. The scene should feel dense and slightly chaotic — human activity at scale.

Global style (applied to all tiles, do not repeat in tile prompts): aerial photography, hyperrealistic, soft midday light, shallow depth of field, Hasselblad medium format

Imagine the finished artwork completely in your mind. Then describe what you would see in each tile.

Rules:
- Each tile prompt must be self-contained. Do not reference other tiles.
- Describe only what is visible within that tile's region of the frame.
- Be specific about local content: what object, what action, what light condition exists in this exact region.
- Maintain consistent vantage point, lighting direction, and atmosphere across all tiles.
- Do not repeat the global style tokens — those are handled separately.
- Tile position (x, y) is 1-indexed. x = column (left to right), y = row (top to bottom).

Return a JSON array with exactly 24 objects. No explanation, no preamble, no markdown fences. Raw JSON only.

Format:
[
  {
    "position": { "x": 1, "y": 1 },
    "prompt": "..."
  },
  ...
]
```

---

## Notes

**Global positive vs. tile prompts.** Style tokens, medium, lighting register — put these in `global_positive` in ComfyUI, not in the tile prompts. The node appends `global_positive` to every tile at generation time. Repeating those tokens in each tile prompt doubles them and wastes context.

**Prompt length per tile.** 30–80 words per tile is the target range. Enough to specify local content precisely. Not so much that you're describing the whole image again.

**Concept types that work well with this method:**
- Scenes with natural regional variation (crowds, cityscapes, landscapes, interiors)
- Compositions with a clear vantage point that holds across the whole image (aerial, straight-on elevation, isometric)
- Subject matter where local detail matters — the tile-level specificity is where the complexity comes from

**Concept types that need extra care:**
- Single subjects that span the whole frame — the LLM needs to think about which body part or detail occupies each tile
- Strong geometric compositions — describe the geometry explicitly per tile or it will drift

**SeamlessX.** If `seamlessX` is enabled, the rightmost column wraps into the leftmost. Tell the LLM this in the master concept if you want the wrap to be intentional: *"the composition should repeat seamlessly left to right — the right edge of the last column should match the left edge of the first."*
