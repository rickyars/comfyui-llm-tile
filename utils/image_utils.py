import numpy as np
import torch
import torch.nn.functional as F


def resize_mask_to_latent(outpaint_mask, latent_h, latent_w, device=None):
    """Resize outpaint_mask [1, H, W, 1] to latent space [1, latent_H, latent_W]."""
    mask = outpaint_mask[:, :, :, 0]
    result = F.interpolate(mask.unsqueeze(1), size=(latent_h, latent_w), mode='nearest').squeeze(1)
    return result.to(device) if device is not None else result


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
        if has_left and overlap_x > 0:
            matched_left = tile_cpu[start_y:start_y + tile_height, 0:overlap_x, :]
            alpha = torch.linspace(0.0, 1.0, overlap_x).view(1, overlap_x, 1)
            zone = canvas[0, pos_y:pos_y + tile_height, pos_x - overlap_x:pos_x, :].clone()
            canvas[0, pos_y:pos_y + tile_height, pos_x - overlap_x:pos_x, :] = (
                (1.0 - alpha) * zone + alpha * matched_left
            )

        if has_top and overlap_y > 0:
            matched_top = tile_cpu[0:overlap_y, start_x:start_x + tile_width, :]
            alpha = torch.linspace(0.0, 1.0, overlap_y).view(overlap_y, 1, 1)
            zone = canvas[0, pos_y - overlap_y:pos_y, pos_x:pos_x + tile_width, :].clone()
            canvas[0, pos_y - overlap_y:pos_y, pos_x:pos_x + tile_width, :] = (
                (1.0 - alpha) * zone + alpha * matched_top
            )

        if has_left and has_top and overlap_x > 0 and overlap_y > 0:
            matched_corner = tile_cpu[0:overlap_y, 0:overlap_x, :]
            alpha_x = torch.linspace(0.0, 1.0, overlap_x).view(1, overlap_x, 1).expand(overlap_y, -1, 1)
            alpha_y = torch.linspace(0.0, 1.0, overlap_y).view(overlap_y, 1, 1).expand(-1, overlap_x, 1)
            alpha = torch.min(alpha_x, alpha_y)
            zone = canvas[0, pos_y - overlap_y:pos_y, pos_x - overlap_x:pos_x, :].clone()
            canvas[0, pos_y - overlap_y:pos_y, pos_x - overlap_x:pos_x, :] = (
                (1.0 - alpha) * zone + alpha * matched_corner
            )

    canvas[0, pos_y:pos_y + tile_height, pos_x:pos_x + tile_width, :] = extracted


def gaussian_blend_tiles(tiles, positions, tile_width, tile_height, overlap_x, overlap_y, final_width, final_height,
                         sigma=0.4):
    """
    Blend tiles using Gaussian weights after all tiles have been generated.

    Args:
        tiles: List of image tensors in any shape
        positions: List of (x, y) positions for each tile
        tile_width/height: Dimensions of each tile
        overlap_x/y: Overlap amounts in pixels
        final_width/height: Final image dimensions
        sigma: Controls the spread of the Gaussian (smaller = sharper transition)

    Returns:
        Blended image tensor [1, H, W, C]
    """
    # Determine device from the first tile
    device = tiles[0].device if len(tiles) > 0 else torch.device('cpu')

    # Create empty canvas and weight accumulator
    final_tensor = torch.zeros((1, final_height, final_width, 3), dtype=torch.float32, device=device)
    weight_accumulator = torch.zeros((1, final_height, final_width, 1), dtype=torch.float32, device=device)

    # Process each tile
    for i, (original_tile, (pos_x, pos_y)) in enumerate(zip(tiles, positions)):
        # Print debug info
        #print(f"Processing tile {i}, shape: {original_tile.shape}, position: ({pos_x}, {pos_y})")

        # Calculate valid region (handle edge cases)
        h = min(tile_height, final_height - pos_y)
        w = min(tile_width, final_width - pos_x)

        # Create weight map for this tile
        weights = np.ones((h, w), dtype=np.float32)

        # Apply Gaussian falloff in overlap regions
        # Left edge (if not leftmost tile)
        if pos_x > 0 and overlap_x > 0:
            for x in range(min(overlap_x, w)):
                # Distance from edge (0 at edge, 1 at overlap boundary)
                dist = x / overlap_x
                weights[:, x] *= np.exp(-((1 - dist) ** 2) / (2 * sigma ** 2))

        # Right edge (if not rightmost tile)
        if pos_x + w < final_width and overlap_x > 0:
            for x in range(max(0, w - overlap_x), w):
                dist = (w - 1 - x) / overlap_x
                weights[:, x] *= np.exp(-((1 - dist) ** 2) / (2 * sigma ** 2))

        # Top edge (if not topmost tile)
        if pos_y > 0 and overlap_y > 0:
            for y in range(min(overlap_y, h)):
                dist = y / overlap_y
                weights[y, :] *= np.exp(-((1 - dist) ** 2) / (2 * sigma ** 2))

        # Bottom edge (if not bottommost tile)
        if pos_y + h < final_height and overlap_y > 0:
            for y in range(max(0, h - overlap_y), h):
                dist = (h - 1 - y) / overlap_y
                weights[y, :] *= np.exp(-((1 - dist) ** 2) / (2 * sigma ** 2))

        # Convert weights to tensor with same shape as tile region
        weights_tensor = torch.from_numpy(weights).to(device).reshape(h, w, 1)

        try:
            # Handle different tensor shapes
            if len(original_tile.shape) == 4:  # [B, H, W, C]
                # Extract the first element from batch
                tile_section = original_tile[0, :h, :w, :]
            elif len(original_tile.shape) == 3:  # [H, W, C]
                # No batch dimension
                tile_section = original_tile[:h, :w, :]
            else:
                print(f"Unexpected tensor shape: {original_tile.shape}")
                continue

            # Apply weighted accumulation directly with matched shapes
            weighted_tile = tile_section * weights_tensor

            # Add to the final tensor
            final_tensor[0, pos_y:pos_y + h, pos_x:pos_x + w, :] += weighted_tile
            weight_accumulator[0, pos_y:pos_y + h, pos_x:pos_x + w, :] += weights_tensor

        except Exception as e:
            print(f"Error processing tile {i}: {e}")
            print(f"Tile shape: {original_tile.shape}")
            print(f"Weights shape: {weights_tensor.shape}")
            print(f"Position: ({pos_x}, {pos_y}), size: ({w}, {h})")
            raise

    # Normalize by accumulated weights (with small epsilon to avoid division by zero)
    epsilon = 1e-8
    mask = weight_accumulator > epsilon
    final_tensor = torch.where(mask, final_tensor / weight_accumulator, final_tensor)

    return final_tensor
