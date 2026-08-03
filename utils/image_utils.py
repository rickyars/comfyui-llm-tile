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


def feather_blend_latent(canvas, refined, y1, x1, overlap_l, has_left, has_top,
                         overlap_x=None, overlap_y=None):
    """
    Write a refined latent tile into canvas with linear feathering on overlap edges.

    canvas:   [B, C, H, W] CPU tensor being assembled in-place
    refined:  [B, C, tile_h, tile_w] sampler output (moved to CPU internally)
    y1, x1:   top-left insertion corner in canvas coordinates
    overlap_l: overlap width/height in latent pixels
    has_left:  True when a previously placed tile overlaps from the left
    has_top:   True when a previously placed tile overlaps from above
    overlap_x/overlap_y: actual per-axis overlap with the neighbor, when it
      differs from overlap_l. The clamped last tile per axis can overlap its
      neighbor by more than the configured overlap (even when overlap_l == 0),
      and hard-writing that zone unfeathered leaves a seam — so the feather
      width must follow the real overlap, not the configured one.
    """
    _, _, tile_h, tile_w = refined.shape
    refined_cpu = refined.cpu()
    ox = overlap_l if overlap_x is None else overlap_x
    oy = overlap_l if overlap_y is None else overlap_y

    # Save existing canvas values in overlap zones before overwriting
    left_zone = (canvas[:, :, y1:y1 + tile_h, x1:x1 + ox].clone()
                 if (has_left and ox > 0) else None)
    top_zone = (canvas[:, :, y1:y1 + oy, x1:x1 + tile_w].clone()
                if (has_top and oy > 0) else None)
    corner_zone = (canvas[:, :, y1:y1 + oy, x1:x1 + ox].clone()
                   if (has_left and has_top and ox > 0 and oy > 0) else None)

    # Hard-write the full refined tile
    canvas[:, :, y1:y1 + tile_h, x1:x1 + tile_w] = refined_cpu

    def _smoothstep(t):
        return t * t * (3.0 - 2.0 * t)

    # Left overlap: ramp alpha 0→1 across overlap columns (old canvas → refined)
    if left_zone is not None and tile_w > ox:
        alpha = _smoothstep(torch.linspace(0.0, 1.0, ox, device=canvas.device)).view(1, 1, 1, ox)
        canvas[:, :, y1:y1 + tile_h, x1:x1 + ox] = (
            (1.0 - alpha) * left_zone + alpha * refined_cpu[:, :, :, :ox]
        )

    # Top overlap: ramp alpha 0→1 across overlap rows (old canvas → refined)
    if top_zone is not None and tile_h > oy:
        alpha = _smoothstep(torch.linspace(0.0, 1.0, oy, device=canvas.device)).view(1, 1, oy, 1)
        canvas[:, :, y1:y1 + oy, x1:x1 + tile_w] = (
            (1.0 - alpha) * top_zone + alpha * refined_cpu[:, :, :oy, :]
        )

    # Corner: min(alpha_x, alpha_y) for smooth 2D diagonal blend
    if corner_zone is not None and tile_w > ox and tile_h > oy:
        alpha_x = _smoothstep(torch.linspace(0.0, 1.0, ox, device=canvas.device)).view(1, 1, 1, ox)
        alpha_y = _smoothstep(torch.linspace(0.0, 1.0, oy, device=canvas.device)).view(1, 1, oy, 1)
        alpha = torch.min(
            alpha_x.expand(1, 1, oy, ox),
            alpha_y.expand(1, 1, oy, ox),
        )
        canvas[:, :, y1:y1 + oy, x1:x1 + ox] = (
            (1.0 - alpha) * corner_zone + alpha * refined_cpu[:, :, :oy, :ox]
        )


def _axis_starts(D, tile_l, overlap_l):
    """
    Full-coverage tile start positions along one axis, MultiDiffusion-style.

    Tiles advance by ``stride = tile_l - overlap_l``; the last tile is clamped
    to end exactly at the canvas edge (``D - tile_l``), overlapping its
    neighbor by more than overlap_l when D is not stride-aligned. Every
    position in [0, D) is covered by at least one tile — there is never an
    uncovered margin, so no pad/crop edge handling is needed.
    """
    if D <= tile_l:
        return [0]
    stride = max(1, tile_l - overlap_l)
    starts = []
    s = 0
    while s + tile_l < D:
        starts.append(s)
        s += stride
    starts.append(D - tile_l)
    return starts


def compute_tile_coords(W, H, tile_l, overlap_l=0):
    """
    Return row-major full-coverage tile coordinates in latent space.

    Returns (coords, n_cols, n_rows) where coords is a list of
    (y1, x1, y2, x2) and the grid is rectangular: coords[r * n_cols + c].
    All tiles are the same size — min(tile_l, axis length) per axis — so the
    model always sees a uniform tile resolution. Coverage is exact: the union
    of tiles equals the whole canvas (see _axis_starts).
    """
    tw = min(tile_l, W)
    th = min(tile_l, H)
    xs = _axis_starts(W, tw, overlap_l)
    ys = _axis_starts(H, th, overlap_l)
    coords = [(y, x, y + th, x + tw) for y in ys for x in xs]
    return coords, len(xs), len(ys)
