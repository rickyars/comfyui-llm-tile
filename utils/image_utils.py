import torch
import torch.nn.functional as F


def blend_and_place_tile(canvas, generated_tile, pos_x, pos_y,
                          tile_width, tile_height, overlap_x, overlap_y,
                          has_left, has_top, controlnet_active):
    """
    Extract the new-content zone from generated_tile, crossfade the ControlNet-matched
    overlap strips into the neighbor's already-placed edge, then hard-place the extracted zone.

    generated_tile shape: [gen_h, gen_w, 3] where gen_h = tile_height + (overlap_y if has_top else 0)
    and gen_w = tile_width + (overlap_x if has_left else 0), when controlnet_active is True.
    Without ControlNet, gen_h = tile_height, gen_w = tile_width (no expansion).
    """
    tile_cpu = generated_tile.cpu() if generated_tile.is_cuda else generated_tile

    start_x = (overlap_x if has_left else 0) if controlnet_active else 0
    start_y = (overlap_y if has_top else 0) if controlnet_active else 0
    extracted = tile_cpu[start_y:start_y + tile_height, start_x:start_x + tile_width, :]

    if controlnet_active:
        device = canvas.device
        if has_left and overlap_x > 0:
            matched_left = tile_cpu[start_y:start_y + tile_height, 0:overlap_x, :]
            alpha = torch.linspace(0.0, 1.0, overlap_x, device=device).view(1, overlap_x, 1)
            zone = canvas[0, pos_y:pos_y + tile_height, pos_x - overlap_x:pos_x, :].clone()
            canvas[0, pos_y:pos_y + tile_height, pos_x - overlap_x:pos_x, :] = (
                (1.0 - alpha) * zone + alpha * matched_left
            )

        if has_top and overlap_y > 0:
            matched_top = tile_cpu[0:overlap_y, start_x:start_x + tile_width, :]
            alpha = torch.linspace(0.0, 1.0, overlap_y, device=device).view(overlap_y, 1, 1)
            zone = canvas[0, pos_y - overlap_y:pos_y, pos_x:pos_x + tile_width, :].clone()
            canvas[0, pos_y - overlap_y:pos_y, pos_x:pos_x + tile_width, :] = (
                (1.0 - alpha) * zone + alpha * matched_top
            )

        if has_left and has_top and overlap_x > 0 and overlap_y > 0:
            matched_corner = tile_cpu[0:overlap_y, 0:overlap_x, :]
            alpha_x = torch.linspace(0.0, 1.0, overlap_x, device=device).view(1, overlap_x, 1)
            alpha_y = torch.linspace(0.0, 1.0, overlap_y, device=device).view(overlap_y, 1, 1)
            alpha = torch.min(alpha_x, alpha_y)  # broadcasts to [overlap_y, overlap_x, 1]
            zone = canvas[0, pos_y - overlap_y:pos_y, pos_x - overlap_x:pos_x, :].clone()
            canvas[0, pos_y - overlap_y:pos_y, pos_x - overlap_x:pos_x, :] = (
                (1.0 - alpha) * zone + alpha * matched_corner
            )

    canvas[0, pos_y:pos_y + tile_height, pos_x:pos_x + tile_width, :] = extracted


def build_working_tensor(final_tensor, final_pos_x, final_pos_y,
                         tile_width, tile_height, overlap_x, overlap_y,
                         gen_w8, gen_h8, has_left_neighbor, has_top_neighbor,
                         wrap_x=False, wrap_y=False):
    """
    Build the expanded generation canvas for one tile from already-placed neighbours.

    Copies the left/top neighbour overlap strips (and the corner), plus the
    seamless wrap strips when the tile is on the far edge, into a fresh
    working tensor, and marks the copied pixels in keep_mask.

    final_tensor: [1, final_H, final_W, 3] canvas being assembled
    gen_w8/gen_h8: 8-aligned generation canvas size for this tile
    wrap_x/wrap_y: True when this tile borders the seamless wrap edge

    Returns (working_tensor [1, gen_h8, gen_w8, 3], keep_mask [gen_h8, gen_w8])
    where keep_mask is 1 wherever neighbour pixels were copied in.
    """
    _, final_height, final_width, _ = final_tensor.shape
    working_tensor = torch.zeros((1, gen_h8, gen_w8, 3), dtype=torch.float32)
    keep_mask = torch.zeros((gen_h8, gen_w8), dtype=torch.float32)

    if has_left_neighbor:
        source_x = final_pos_x - overlap_x
        source_start_y = final_pos_y
        source_end_y = min(final_pos_y + tile_height, final_height)
        source_height = source_end_y - source_start_y
        target_start_y = overlap_y if has_top_neighbor else 0
        copy_height = min(source_height, gen_h8 - target_start_y)
        if copy_height > 0:
            working_tensor[0,
                target_start_y:target_start_y + copy_height, :overlap_x, :
            ] = final_tensor[0,
                source_start_y:source_start_y + copy_height,
                source_x:source_x + overlap_x, :]
            keep_mask[target_start_y:target_start_y + copy_height, :overlap_x] = 1.0

    if wrap_x and overlap_x > 0:
        wrap_target_x = overlap_x + tile_width
        source_start_y = final_pos_y
        source_end_y = min(final_pos_y + tile_height, final_height)
        source_height = source_end_y - source_start_y
        target_start_y = overlap_y if has_top_neighbor else 0
        copy_height = min(source_height, gen_h8 - target_start_y)
        if copy_height > 0 and wrap_target_x + overlap_x <= gen_w8:
            working_tensor[0,
                target_start_y:target_start_y + copy_height,
                wrap_target_x:wrap_target_x + overlap_x, :
            ] = final_tensor[0,
                source_start_y:source_start_y + copy_height, 0:overlap_x, :]
            keep_mask[target_start_y:target_start_y + copy_height,
                      wrap_target_x:wrap_target_x + overlap_x] = 1.0

    if wrap_y and overlap_y > 0:
        wrap_target_y = overlap_y + tile_height
        source_start_x = final_pos_x
        source_end_x = min(final_pos_x + tile_width, final_width)
        source_width = source_end_x - source_start_x
        target_start_x = overlap_x if has_left_neighbor else 0
        copy_width = min(source_width, gen_w8 - target_start_x)
        if copy_width > 0 and wrap_target_y + overlap_y <= gen_h8:
            working_tensor[0,
                wrap_target_y:wrap_target_y + overlap_y,
                target_start_x:target_start_x + copy_width, :
            ] = final_tensor[0,
                0:overlap_y, source_start_x:source_start_x + copy_width, :]
            keep_mask[wrap_target_y:wrap_target_y + overlap_y,
                      target_start_x:target_start_x + copy_width] = 1.0

    if has_top_neighbor:
        source_y = final_pos_y - overlap_y
        source_start_x = final_pos_x
        source_end_x = min(final_pos_x + tile_width, final_width)
        source_width = source_end_x - source_start_x
        target_start_x = overlap_x if has_left_neighbor else 0
        copy_width = min(source_width, gen_w8 - target_start_x)
        if copy_width > 0:
            working_tensor[0,
                :overlap_y, target_start_x:target_start_x + copy_width, :
            ] = final_tensor[0,
                source_y:source_y + overlap_y,
                source_start_x:source_start_x + copy_width, :]
            keep_mask[:overlap_y, target_start_x:target_start_x + copy_width] = 1.0

    if has_left_neighbor and has_top_neighbor:
        corner_source_x = final_pos_x - overlap_x
        corner_source_y = final_pos_y - overlap_y
        working_tensor[0, :overlap_y, :overlap_x, :] = final_tensor[0,
            corner_source_y:corner_source_y + overlap_y,
            corner_source_x:corner_source_x + overlap_x, :]
        keep_mask[:overlap_y, :overlap_x] = 1.0

    return working_tensor, keep_mask


def feather_blend_latent(canvas, refined, y1, x1, overlap_l, has_left, has_top):
    """
    Write a refined latent tile into canvas with linear feathering on overlap edges.

    canvas:   [B, C, H, W] CPU tensor being assembled in-place
    refined:  [B, C, tile_h, tile_w] sampler output (moved to CPU internally)
    y1, x1:   top-left insertion corner in canvas coordinates
    overlap_l: overlap width/height in latent pixels
    has_left:  True when a previously placed tile overlaps from the left
    has_top:   True when a previously placed tile overlaps from above
    """
    _, _, tile_h, tile_w = refined.shape
    refined_cpu = refined.cpu()

    # Save existing canvas values in overlap zones before overwriting
    left_zone = (canvas[:, :, y1:y1 + tile_h, x1:x1 + overlap_l].clone()
                 if (has_left and overlap_l > 0) else None)
    top_zone = (canvas[:, :, y1:y1 + overlap_l, x1:x1 + tile_w].clone()
                if (has_top and overlap_l > 0) else None)
    corner_zone = (canvas[:, :, y1:y1 + overlap_l, x1:x1 + overlap_l].clone()
                   if (has_left and has_top and overlap_l > 0) else None)

    # Hard-write the full refined tile
    canvas[:, :, y1:y1 + tile_h, x1:x1 + tile_w] = refined_cpu

    def _smoothstep(t):
        return t * t * (3.0 - 2.0 * t)

    # Left overlap: ramp alpha 0→1 across overlap columns (old canvas → refined)
    if left_zone is not None and tile_w > overlap_l:
        alpha = _smoothstep(torch.linspace(0.0, 1.0, overlap_l, device=canvas.device)).view(1, 1, 1, overlap_l)
        canvas[:, :, y1:y1 + tile_h, x1:x1 + overlap_l] = (
            (1.0 - alpha) * left_zone + alpha * refined_cpu[:, :, :, :overlap_l]
        )

    # Top overlap: ramp alpha 0→1 across overlap rows (old canvas → refined)
    if top_zone is not None and tile_h > overlap_l:
        alpha = _smoothstep(torch.linspace(0.0, 1.0, overlap_l, device=canvas.device)).view(1, 1, overlap_l, 1)
        canvas[:, :, y1:y1 + overlap_l, x1:x1 + tile_w] = (
            (1.0 - alpha) * top_zone + alpha * refined_cpu[:, :, :overlap_l, :]
        )

    # Corner: min(alpha_x, alpha_y) for smooth 2D diagonal blend
    if corner_zone is not None and tile_w > overlap_l and tile_h > overlap_l:
        alpha_x = _smoothstep(torch.linspace(0.0, 1.0, overlap_l, device=canvas.device)).view(1, 1, 1, overlap_l)
        alpha_y = _smoothstep(torch.linspace(0.0, 1.0, overlap_l, device=canvas.device)).view(1, 1, overlap_l, 1)
        alpha = torch.min(
            alpha_x.expand(1, 1, overlap_l, overlap_l),
            alpha_y.expand(1, 1, overlap_l, overlap_l),
        )
        canvas[:, :, y1:y1 + overlap_l, x1:x1 + overlap_l] = (
            (1.0 - alpha) * corner_zone + alpha * refined_cpu[:, :, :overlap_l, :overlap_l]
        )


def _compute_center_grid(W, H, tile_l, overlap_l):
    """
    Compute a full-coverage tile grid. Grid count is determined by tile_l only.

    Returns (cols, rows) — number of strides in each axis.
    Tile count in each axis is cols+1 / rows+1.

    Tile positions are anchored at multiples of tile_l. When overlap_l > 0,
    each non-edge tile extends into its neighbor's territory by overlap_l pixels
    (see _compute_tile_coords), so the grid count stays the same but tiles overlap.
    """
    cols = max(0, W // tile_l - 1)
    rows = max(0, H // tile_l - 1)
    return cols, rows


def _axis_pad(D, tile_l):
    """Padding (before, after) for one axis so the centered grid is preserved.

    The centered grid (see _compute_center_grid / _compute_tile_coords) places
    ``n = D // tile_l`` whole tiles with an undetailed margin of ``round(rem/2)``
    on the near side and the rest on the far side, where ``rem = D % tile_l``.
    To cover those margins *without moving any existing tile*, we pad each side
    that has a margin by ``tile_l - margin`` — turning the partial edge strip
    into one extra full tile while every original tile keeps its coordinate.
    The result is always an exact multiple of tile_l.
    """
    if D <= tile_l:
        # Smaller than one tile: centre it inside a single tile.
        total = tile_l - D
        return total // 2, total - total // 2
    rem = D % tile_l
    if rem == 0:
        return 0, 0
    near_margin = round(rem / 2)          # matches _compute_tile_coords start
    far_margin = rem - near_margin
    pad_before = (tile_l - near_margin) if near_margin > 0 else 0
    pad_after = (tile_l - far_margin) if far_margin > 0 else 0
    return pad_before, pad_after


def pad_latent_to_grid(canvas, tile_l):
    """Edge-replicate pad a latent canvas so the centered grid covers it fully.

    Returns (padded_canvas, (pad_top, pad_left)). Padding preserves the
    center-mode tile positions and appends one edge tile per uncovered margin
    (see _axis_pad), so pad mode produces the same detail grid as center mode
    plus edge coverage. An axis already a multiple of tile_l receives zero
    padding; if both axes are aligned the original tensor is returned unchanged.
    The (pad_top, pad_left) offsets locate the original region inside the padded
    canvas so callers can crop back afterward.
    """
    _, _, H, W = canvas.shape
    pad_top, pad_bottom = _axis_pad(H, tile_l)
    pad_left, pad_right = _axis_pad(W, tile_l)
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return canvas, (0, 0)
    padded = F.pad(
        canvas, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
    )
    return padded, (pad_top, pad_left)


def _compute_tile_coords(W, H, tile_l, cols, rows, overlap_l=0):
    """
    Return row-major tile coordinates in latent space.

    The core grid is centered: start_x = round((W - (cols+1)*tile_l) / 2).
    Each tile anchor is at (start_x + c*tile_l, start_y + r*tile_l). When
    overlap_l > 0, non-edge tiles are grown into their left/top neighbor's
    territory by overlap_l pixels, creating positional overlap without
    adding extra tiles to the grid.
    """
    start_x = max(0, round((W - (cols + 1) * tile_l) / 2))
    start_y = max(0, round((H - (rows + 1) * tile_l) / 2))
    coords = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            x_anchor = start_x + c * tile_l
            y_anchor = start_y + r * tile_l
            x1 = (x_anchor - overlap_l) if c > 0 else x_anchor
            y1 = (y_anchor - overlap_l) if r > 0 else y_anchor
            x2 = x_anchor + tile_l
            y2 = y_anchor + tile_l
            coords.append((y1, x1, y2, x2))
    return coords
