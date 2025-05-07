import numpy as np
import torch


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
