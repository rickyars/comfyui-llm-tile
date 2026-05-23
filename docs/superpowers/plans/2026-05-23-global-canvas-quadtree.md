# Global Canvas Quadtree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-tile quadtree in `quadtree_density` scoring with a single global canvas quadtree so flat regions stay large via natural heap competition — no threshold parameter required.

**Architecture:** One global greedy max-heap runs over the full canvas (like the JS reference in `E:\projects\pixelator\studio\quadtree-builder.js`). Tiles are scored by counting how many global leaf cells have their center within the tile bounds. Flat regions naturally stay undivided because complex regions always win the heap; budget is auto-scaled from canvas dimensions.

**Tech Stack:** Python, PyTorch, `heapq` (already imported)

---

## File Map

| File | Change |
|---|---|
| `node_detailer_adaptive.py` | Extract `_region_detail`, add `_build_canvas_quadtree`, replace `_tile_quadtree_density` body, replace `_build_quadtree_map` body, update call site |
| `tests/test_adaptive_detailer.py` | Add tests for `_build_canvas_quadtree`, update 1 existing test assertion, rename 1 test |

---

### Task 1: Extract `_region_detail` to module level

**Files:**
- Modify: `node_detailer_adaptive.py`

This is a pure refactor — no behavior change. `_region_detail` is currently an inner closure in both `_tile_quadtree_density` (line ~251) and `_build_quadtree_map` (line ~328), closing over `sample`. Pull it out with `sample` as an explicit argument so `_build_canvas_quadtree` (Task 2) can share it.

- [ ] **Step 1: Add module-level `_region_detail` just above the `_tile_quadtree_density` definition (around line 229)**

Insert this block:

```python
def _region_detail(sample, ry, rx, rh, rw):
    if rh * rw < 2:
        return 0.0
    return sample[:, ry:ry + rh, rx:rx + rw].std(dim=[1, 2]).sum().item()
```

- [ ] **Step 2: Remove the inner `_region_detail` from `_tile_quadtree_density` and update its call sites**

In `_tile_quadtree_density`, delete the inner function definition (currently the first 5 lines of the function body, the closure that reads `sample` from the outer scope). Update every call from `_region_detail(ry, rx, rh, rw)` to `_region_detail(sample, ry, rx, rh, rw)`.

There are two call sites inside the function:
```python
root_detail = _region_detail(sample, y1, x1, th, tw)
```
and:
```python
heapq.heappush(heap, (-_region_detail(sample, cy, cx, ch, cw), cy, cx, ch, cw))
```

- [ ] **Step 3: Remove the inner `_region_detail` from `_build_quadtree_map` and update its call sites**

Same change: delete the inner function definition (5 lines, around line 327), update calls to pass `sample` as the first argument. The two call sites:
```python
root_detail = _region_detail(sample, y1, x1, th, tw)
```
and:
```python
heapq.heappush(heap, (-_region_detail(sample, cy, cx, ch, cw), cy, cx, ch, cw))
```

- [ ] **Step 4: Run existing tests to verify no regression**

```
cd E:\StableDiffusion\ComfyUI\custom_nodes\comfyui-llm-tile
python -m pytest tests/test_adaptive_detailer.py -v -k "quadtree"
```

Expected: all quadtree tests pass (behavior unchanged — same logic, just refactored location).

- [ ] **Step 5: Commit**

```
git add node_detailer_adaptive.py
git commit -m "refactor: extract _region_detail to module level"
```

---

### Task 2: Implement `_build_canvas_quadtree`

**Files:**
- Modify: `node_detailer_adaptive.py`
- Modify: `tests/test_adaptive_detailer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_adaptive_detailer.py` (after the existing quadtree imports):

```python
from node_detailer_adaptive import _build_canvas_quadtree


def test_build_canvas_quadtree_flat_returns_one_leaf():
    # Flat canvas: root detail <= epsilon, stays as the single leaf.
    canvas = torch.zeros(1, 4, 32, 32)
    leaves = _build_canvas_quadtree(canvas)
    assert len(leaves) == 1
    assert leaves[0] == (0, 0, 32, 32)


def test_build_canvas_quadtree_complex_returns_multiple_leaves():
    torch.manual_seed(42)
    canvas = torch.randn(1, 4, 32, 32)
    leaves = _build_canvas_quadtree(canvas)
    assert len(leaves) > 1


def test_build_canvas_quadtree_leaves_partition_canvas():
    # Leaves must cover exactly H*W latent cells (no gaps, no overlaps).
    torch.manual_seed(42)
    canvas = torch.randn(1, 4, 32, 32)
    leaves = _build_canvas_quadtree(canvas)
    total_area = sum(rh * rw for (ry, rx, rh, rw) in leaves)
    assert total_area == 32 * 32


def test_build_canvas_quadtree_complex_region_gets_more_leaves():
    # Bottom-right quadrant is complex; top-left is flat.
    # Global heap spends budget on the complex region.
    torch.manual_seed(0)
    canvas = torch.zeros(1, 4, 32, 32)
    canvas[:, :, 16:32, 16:32] = torch.randn(1, 4, 16, 16)
    leaves = _build_canvas_quadtree(canvas)
    flat_leaves = [l for l in leaves if l[0] < 16 and l[1] < 16]
    complex_leaves = [l for l in leaves if l[0] >= 16 and l[1] >= 16]
    assert len(complex_leaves) > len(flat_leaves)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/test_adaptive_detailer.py -v -k "build_canvas_quadtree"
```

Expected: `ImportError` or `NameError` — `_build_canvas_quadtree` does not exist yet.

- [ ] **Step 3: Implement `_build_canvas_quadtree` in `node_detailer_adaptive.py`**

Insert immediately after the `_region_detail` function you added in Task 1 (before `_tile_quadtree_density`):

```python
def _build_canvas_quadtree(canvas, min_cell=4, max_iterations=None):
    """
    Run a single greedy quadtree over the whole canvas.

    Always splits the highest-detail region first (max-heap). Flat regions
    never rise to the top of the heap in the presence of complex regions, so
    they naturally stay as large cells — no threshold required.

    Stopping conditions:
      - max_iterations budget exhausted (auto-scaled from canvas size if None)
      - detail <= _QUIET_SCORE_EPSILON (genuinely flat cell)
      - cell smaller than min_cell in either dimension

    Returns: list of (ry, rx, rh, rw) leaf cells covering the full canvas.
    """
    sample = canvas[0]  # [C, H, W]
    _, H, W = sample.shape

    if max_iterations is None:
        max_iterations = (H // min_cell) * (W // min_cell) // 2

    root_detail = _region_detail(sample, 0, 0, H, W)
    heap = [(-root_detail, 0, 0, H, W)]
    leaves = []

    for _ in range(max_iterations):
        if not heap:
            break

        neg_d, ry, rx, rh, rw = heapq.heappop(heap)
        d = -neg_d

        if d <= _QUIET_SCORE_EPSILON:
            leaves.append((ry, rx, rh, rw))
            continue

        half_h = rh // 2
        half_w = rw // 2
        can_h = half_h >= min_cell
        can_w = half_w >= min_cell

        if not can_h and not can_w:
            leaves.append((ry, rx, rh, rw))
            continue

        if can_h and can_w:
            children = [
                (ry,          rx,           half_h,       half_w),
                (ry,          rx + half_w,  half_h,       rw - half_w),
                (ry + half_h, rx,           rh - half_h,  half_w),
                (ry + half_h, rx + half_w,  rh - half_h,  rw - half_w),
            ]
        elif can_h:
            children = [
                (ry,          rx,  half_h,      rw),
                (ry + half_h, rx,  rh - half_h, rw),
            ]
        else:
            children = [
                (ry, rx,          rh,  half_w),
                (ry, rx + half_w, rh,  rw - half_w),
            ]

        for cy, cx, ch, cw in children:
            heapq.heappush(heap, (-_region_detail(sample, cy, cx, ch, cw), cy, cx, ch, cw))

    # Budget exhausted — flush remaining heap entries as leaves
    while heap:
        _, ry, rx, rh, rw = heapq.heappop(heap)
        leaves.append((ry, rx, rh, rw))

    return leaves
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest tests/test_adaptive_detailer.py -v -k "build_canvas_quadtree"
```

Expected: all 4 new tests PASS.

- [ ] **Step 5: Commit**

```
git add node_detailer_adaptive.py tests/test_adaptive_detailer.py
git commit -m "feat: add _build_canvas_quadtree — single global greedy quadtree"
```

---

### Task 3: Replace `_tile_quadtree_density` and update its tests

**Files:**
- Modify: `node_detailer_adaptive.py`
- Modify: `tests/test_adaptive_detailer.py`

The current implementation runs a separate heap per tile. Replace it with a thin wrapper that calls `_build_canvas_quadtree` once and counts leaf centers per tile.

- [ ] **Step 1: Update the three affected tests before touching the implementation**

In `tests/test_adaptive_detailer.py`:

**Rename and update assertion** for the flat-canvas test (currently `test_tile_quadtree_density_flat_canvas_returns_single_leaf`):

```python
def test_tile_quadtree_density_flat_canvas_scores_zero():
    # Global tree: flat canvas → 1 leaf spanning the whole canvas.
    # Its center (16,16) is outside the queried tile (0,0,16,16), so score = 0.
    canvas = torch.zeros(1, 4, 32, 32)
    coords = [(0, 0, 16, 16)]
    result = _tile_quadtree_density(canvas, coords)
    assert result[0] == pytest.approx(0.0)
```

Delete the old `test_tile_quadtree_density_flat_canvas_returns_single_leaf` definition entirely.

The other two tests (`test_tile_quadtree_density_complex_higher_than_flat` and `test_tile_quadtree_density_ranks_tiles_correctly`) keep their existing assertions — verify they still hold after the implementation change (Step 4).

- [ ] **Step 2: Run the updated test to confirm it fails (old implementation)**

```
python -m pytest tests/test_adaptive_detailer.py::test_tile_quadtree_density_flat_canvas_scores_zero -v
```

Expected: FAIL — old implementation returns `1/(16*16)`, not `0.0`.

- [ ] **Step 3: Replace `_tile_quadtree_density` body in `node_detailer_adaptive.py`**

Replace the entire function (keep the name and outer signature `canvas, tile_coords, min_cell=4` — the tests call it this way):

```python
def _tile_quadtree_density(canvas, tile_coords, min_cell=4):
    """
    Score tiles by quadtree leaf density from a single global canvas quadtree.

    Runs one greedy quadtree over the whole canvas (see _build_canvas_quadtree),
    then scores each tile by counting leaves whose center falls within it:

        score = leaves_with_center_in_tile / tile_area

    Flat tiles score near zero (their regions stay as large leaves whose centers
    rarely land inside a particular queried tile). Complex tiles score higher
    (many small leaves concentrated in detailed regions).
    """
    leaves = _build_canvas_quadtree(canvas, min_cell)
    result = []
    for (y1, x1, y2, x2) in tile_coords:
        th = y2 - y1
        tw = x2 - x1
        if th <= 0 or tw <= 0:
            result.append(0.0)
            continue
        count = sum(
            1 for (ry, rx, rh, rw) in leaves
            if y1 <= ry + rh // 2 < y2 and x1 <= rx + rw // 2 < x2
        )
        result.append(count / (th * tw))
    return result
```

- [ ] **Step 4: Run all three tile-density tests**

```
python -m pytest tests/test_adaptive_detailer.py -v -k "tile_quadtree_density"
```

Expected: all three PASS:
- `test_tile_quadtree_density_flat_canvas_scores_zero` — score = 0.0 ✓
- `test_tile_quadtree_density_complex_higher_than_flat` — complex > 0 = flat ✓
- `test_tile_quadtree_density_ranks_tiles_correctly` — complex tile > flat tile ✓

If `test_tile_quadtree_density_complex_higher_than_flat` fails (unlikely but possible if seed 42 produces edge case), add `torch.manual_seed(42)` to the complex canvas setup and check that at least one leaf center falls in `(0, 0, 16, 16)`.

- [ ] **Step 5: Run the full test suite to check for regressions**

```
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```
git add node_detailer_adaptive.py tests/test_adaptive_detailer.py
git commit -m "feat: replace per-tile quadtree with global canvas quadtree scoring"
```

---

### Task 4: Replace `_build_quadtree_map` and clean up call site

**Files:**
- Modify: `node_detailer_adaptive.py`

`_build_quadtree_map` currently runs its own separate quadtree. Replace it with a call to `_build_canvas_quadtree`. The visualization output (white cell outlines on black) is unchanged; the tree that drives it is now the same global tree used for scoring.

Also remove the now-unused `tile_coords` parameter and update the one call site in `detail()`.

- [ ] **Step 1: Replace `_build_quadtree_map` body**

```python
def _build_quadtree_map(canvas, min_cell=4):
    """
    Build a pixel-space visualization of the global canvas quadtree.

    Draws white cell outlines on a dark background using the same global tree
    as _tile_quadtree_density. Large cells = flat regions. Small cells = detail.

    Returns: IMAGE tensor [1, H*8, W*8, 3]
    """
    sample = canvas[0]
    _, H, W = sample.shape
    img = torch.zeros(1, H * 8, W * 8, 3)

    for (ry, rx, rh, rw) in _build_canvas_quadtree(canvas, min_cell):
        py0, py1 = ry * 8, (ry + rh) * 8
        px0, px1 = rx * 8, (rx + rw) * 8
        img[0, py0:min(py0 + 2, py1), px0:px1, :] = 1.0
        img[0, max(py1 - 2, py0):py1, px0:px1, :] = 1.0
        img[0, py0:py1, px0:min(px0 + 2, px1), :] = 1.0
        img[0, py0:py1, max(px1 - 2, px0):px1, :] = 1.0

    return img
```

- [ ] **Step 2: Update the call site in `detail()` (remove `tile_coords` argument)**

Find the existing call (around line 461):
```python
scoring_map_img = _build_quadtree_map(canvas, tile_coords)
```

Replace with:
```python
scoring_map_img = _build_quadtree_map(canvas)
```

- [ ] **Step 3: Run full test suite**

```
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```
git add node_detailer_adaptive.py
git commit -m "refactor: rebuild _build_quadtree_map from global canvas quadtree"
```

---

### Task 5: Final verification

**Files:** none

- [ ] **Step 1: Run full test suite one final time**

```
python -m pytest tests/ -v
```

Expected: all tests pass with no warnings about deprecated parameters.

- [ ] **Step 2: Confirm `detail_fraction` is gone**

```
python -c "import ast, sys; src=open('node_detailer_adaptive.py').read(); print('FOUND' if 'detail_fraction' in src else 'OK')"
```

Expected: `OK`. (The parameter no longer exists anywhere in the module.)

- [ ] **Step 3: Verify the node loads in Python**

```
python -c "from node_detailer_adaptive import LLMAdaptiveTileDetailer; print('OK')"
```

Expected: `OK`
