# Design: Global Canvas Quadtree for Tile Scoring

**Date:** 2026-05-23  
**Branch:** feat/tile-detailer  
**File:** `node_detailer_adaptive.py`

## Problem

`_tile_quadtree_density` runs a separate greedy heap per tile. A tile covering only a flat region (sky, background) competes only against other flat sub-regions — they all have equal-but-nonzero detail, so the heap subdivides them uniformly to `min_cell`. The result is a regular grid regardless of content.

This is also the root cause of the parameter problem: `detail_fraction` attempts to create a threshold that separates "flat" from "complex," but no single fraction works across all images.

## Reference

`E:\projects\pixelator\studio\quadtree-builder.js` — a proven working implementation. The JS builder runs a single greedy heap over the **whole image**. Flat regions stay large because complex regions always outbid them for the iteration budget. No threshold is needed; the stopping condition is `maxIterations` (a budget) plus a minimum cell size guard.

## Approach: Single Global Canvas Quadtree

Run one greedy quadtree over the entire canvas using a max-heap ordered by `_region_detail`. Score each ComfyUI tile by counting how many global leaf cells fall within it, divided by tile area. The visualization draws all global leaves as cell outlines.

This recovers the JS global-competition property: flat regions never rise to the top of the heap in the presence of complex regions, so they stay as large cells naturally. No threshold or fraction parameter.

## Stopping Conditions (two, like JS)

1. **Iteration budget** (`max_iterations`): auto-scaled to `(H // min_cell) * (W // min_cell) // 2` if not specified. Spends budget proportional to how many min-cell-sized leaves the canvas can hold.
2. **Epsilon guard**: if a node's detail ≤ `_QUIET_SCORE_EPSILON`, it stays as a leaf (handles truly flat latents without wasted computation).
3. **Min cell size**: same as current — no splitting below `min_cell` in either dimension.

## New Functions

### `_region_detail(sample, ry, rx, rh, rw)` — module level

Extract the inner function currently duplicated in both `_tile_quadtree_density` and `_build_quadtree_map`. Signature takes `sample: [C, H, W]` explicitly.

### `_build_canvas_quadtree(canvas, min_cell=4, max_iterations=None)` → `list[(ry, rx, rh, rw)]`

Runs the single global greedy quadtree. Returns the flat list of all leaf cells. Remaining heap entries when the budget is exhausted are also leaves.

## Changed Functions

### `_tile_quadtree_density(canvas, tile_coords, min_cell=4)` — new implementation

Calls `_build_canvas_quadtree` internally, then scores each tile by counting leaf centers within its bounds:

```
score = leaves_with_center_in_tile / tile_area
```

Signature is backward-compatible (canvas + tile_coords), so existing tests keep working structurally.

### `_build_quadtree_map(canvas, tile_coords, min_cell=4)` — new implementation

Calls `_build_canvas_quadtree` with the same parameters, draws white outlines for every leaf cell on a black background. Visualization is unchanged in appearance.

## Call Site in `detail()`

Both functions already called in sequence from `detail()`. No change to the call site. The global tree is built twice (once for scoring, once for the map) — acceptable given it's cheap and keeps the functions self-contained. If performance is a concern later, we can cache the leaves and pass them explicitly.

## Test Changes

Three existing quadtree tests need updating:

**`test_tile_quadtree_density_flat_canvas_returns_single_leaf`**  
Old assertion: `result[0] == 1 / (16 * 16)`. In the global approach, the single root leaf spans the full 32×32 canvas; its center falls outside the 16×16 tile, so the tile score is 0.  
New assertion: `result[0] == pytest.approx(0.0)` — flat canvas, no leaf centers in tile, score = 0 = denoise_min. Semantically correct.

**`test_tile_quadtree_density_complex_higher_than_flat`**  
Currently uses a fully-randn canvas. In the global approach, a uniform-randn canvas still has all regions at similar detail, so the heap subdivides somewhat arbitrarily. Replace with a canvas that has a clearly flat half and a clearly complex half (matching `test_tile_quadtree_density_ranks_tiles_correctly`), testing the complex tile against the flat tile.

**`test_tile_quadtree_density_ranks_tiles_correctly`**  
Structurally valid — canvas has zeros in top-left, randn in bottom-right. Global heap naturally prioritizes the complex quadrant. No logical change needed, but verify assertion values after implementation.

## What Does Not Change

- All other scoring methods: `otsu_threshold`, `gradient_magnitude`
- `_build_quadtree_map` output appearance: white cell outlines on black
- Node inputs: no new parameters exposed
- `_build_denoise_map`, `_scores_to_denoise`, `_smooth_scores`, `_t_to_rgb`
