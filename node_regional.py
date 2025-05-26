import torch
import numpy as np
import comfy.sample
from comfy.utils import ProgressBar

from .utils import parse_tile_prompts
from .utils import combine_guider_conditioning, restore_guider_conditioning


class RegionalTiledDiffusion:
    """
    Regional Tiled Diffusion following TiledDiffusion methodology:
    - Fixed pixel overlap (not percent)
    - Gaussian weighted blending in overlap regions
    - Proper img2img with denoise_strength and steps
    - Color consistency through shared normalization
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Input image
                "input_image": ("IMAGE",),

                # Tile configuration
                "json_tile_prompts": ("STRING", {"multiline": True}),
                "grid_width": ("INT", {"default": 4, "min": 1, "max": 16}),
                "grid_height": ("INT", {"default": 6, "min": 1, "max": 16}),
                "tile_width": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "tile_height": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "overlap": ("INT", {"default": 32, "min": 16, "max": 128, "step": 16}),

                # Sampling parameters
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                # Required inputs
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("refined_image",)
    FUNCTION = "process_regional_tiles"
    CATEGORY = "image/generation"

    def process_regional_tiles(self, input_image, json_tile_prompts, grid_width, grid_height,
                               tile_width, tile_height, overlap, seed,
                               noise, guider, sampler, sigmas, clip, vae):
        """Process tiles with TiledDiffusion methodology"""

        # Parse tile prompts
        tile_prompts = parse_tile_prompts(json_tile_prompts, grid_width, grid_height)

        # Get input image dimensions [B, H, W, C]
        input_tensor = input_image[0]  # Remove batch dimension
        img_height, img_width = input_tensor.shape[0], input_tensor.shape[1]
        device = input_tensor.device

        # Calculate tile positions with overlap
        tile_positions = self._calculate_tile_positions(
            img_width, img_height, grid_width, grid_height,
            tile_width, tile_height, overlap
        )

        # Create output canvas and weight accumulator for averaging
        output_canvas = torch.zeros_like(input_tensor, device=device)
        weight_canvas = torch.zeros(img_height, img_width, device=device)

        # Process each tile
        pbar = ProgressBar(len(tile_positions))
        for i, (pos, prompt) in enumerate(zip(tile_positions, tile_prompts)):
            # Extract tile with padding
            tile_tensor = self._extract_tile(input_tensor, pos, tile_width, tile_height)

            # Process tile with regional prompt
            processed_tile = self._process_tile_img2img(
                tile_tensor, prompt, seed + i,
                guider, sampler, sigmas, clip, vae, noise, device
            )

            # Create Gaussian weights for this tile
            weights = self._create_gaussian_weights(tile_width, tile_height, overlap, pos, img_width, img_height)

            # Accumulate weighted tile
            self._accumulate_weighted_tile(output_canvas, weight_canvas, processed_tile, weights, pos)

            pbar.update(1)

        # Normalize by weights to get final result
        # Avoid division by zero
        weight_canvas = torch.clamp(weight_canvas, min=1e-8)
        final_image = output_canvas / weight_canvas.unsqueeze(-1)

        return (final_image.unsqueeze(0),)  # Add batch dimension back

    def _calculate_tile_positions(self, img_width, img_height, grid_width, grid_height,
                                  tile_width, tile_height, overlap):
        """Calculate tile positions with proper overlap"""
        positions = []

        # Calculate step size (distance between tile centers)
        if grid_width > 1:
            step_x = (img_width - tile_width) // (grid_width - 1)
        else:
            step_x = 0

        if grid_height > 1:
            step_y = (img_height - tile_height) // (grid_height - 1)
        else:
            step_y = 0

        for y in range(grid_height):
            for x in range(grid_width):
                if grid_width == 1:
                    pos_x = (img_width - tile_width) // 2
                else:
                    pos_x = x * step_x

                if grid_height == 1:
                    pos_y = (img_height - tile_height) // 2
                else:
                    pos_y = y * step_y

                # Clamp to image boundaries
                pos_x = max(0, min(pos_x, img_width - tile_width))
                pos_y = max(0, min(pos_y, img_height - tile_height))

                positions.append((pos_x, pos_y))

        return positions

    def _extract_tile(self, image_tensor, position, tile_width, tile_height):
        """Extract tile from image tensor"""
        pos_x, pos_y = position
        tile = image_tensor[pos_y:pos_y + tile_height, pos_x:pos_x + tile_width]

        # Ensure exact dimensions (pad if needed at edges)
        if tile.shape[0] != tile_height or tile.shape[1] != tile_width:
            padded_tile = torch.zeros(tile_height, tile_width, 3, device=image_tensor.device, dtype=image_tensor.dtype)
            h, w = tile.shape[0], tile.shape[1]
            padded_tile[:h, :w] = tile
            tile = padded_tile

        return tile

    def _process_tile_img2img(self, tile_tensor, prompt, tile_seed,
                              guider, sampler, sigmas, clip, vae, noise, device):
        """Process single tile with img2img using sigmas schedule"""

        # Add batch dimension and encode to latent
        tile_batch = tile_tensor.unsqueeze(0)  # [1, H, W, C]

        with torch.no_grad():
            latent = vae.encode(tile_batch)

        # Set regional conditioning
        original_conditioning = combine_guider_conditioning(guider, prompt, clip)

        try:
            # Generate noise for img2img
            if noise is not None:
                tile_noise = noise.generate_noise({"samples": latent})
            else:
                tile_noise = comfy.sample.prepare_noise(latent, tile_seed)

            # Sample using the provided sigmas schedule
            refined_latent = guider.sample(
                tile_noise,
                latent,
                sampler,
                sigmas,
                denoise_mask=None,
                disable_pbar=False,  # Show progress
                seed=tile_seed
            )

        finally:
            # Restore original conditioning
            restore_guider_conditioning(guider, original_conditioning)

        # Decode back to image
        with torch.no_grad():
            decoded = vae.decode(refined_latent)

        return decoded[0]  # Remove batch dimension

    def _create_gaussian_weights(self, tile_width, tile_height, overlap, position, img_width, img_height):
        """Create Gaussian weights for tile blending following TiledDiffusion approach"""
        pos_x, pos_y = position
        weights = torch.ones(tile_height, tile_width, device='cpu')

        # Apply Gaussian falloff in overlap regions
        sigma = overlap / 3.0  # Standard deviation for Gaussian

        # Left edge
        if pos_x > 0:
            for x in range(min(overlap, tile_width)):
                dist_from_edge = x / overlap
                weight = np.exp(-(1 - dist_from_edge) ** 2 / (2 * (sigma / overlap) ** 2))
                weights[:, x] *= weight

        # Right edge
        if pos_x + tile_width < img_width:
            for x in range(max(0, tile_width - overlap), tile_width):
                dist_from_edge = (tile_width - 1 - x) / overlap
                weight = np.exp(-(1 - dist_from_edge) ** 2 / (2 * (sigma / overlap) ** 2))
                weights[:, x] *= weight

        # Top edge
        if pos_y > 0:
            for y in range(min(overlap, tile_height)):
                dist_from_edge = y / overlap
                weight = np.exp(-(1 - dist_from_edge) ** 2 / (2 * (sigma / overlap) ** 2))
                weights[y, :] *= weight

        # Bottom edge
        if pos_y + tile_height < img_height:
            for y in range(max(0, tile_height - overlap), tile_height):
                dist_from_edge = (tile_height - 1 - y) / overlap
                weight = np.exp(-(1 - dist_from_edge) ** 2 / (2 * (sigma / overlap) ** 2))
                weights[y, :] *= weight

        return weights

    def _accumulate_weighted_tile(self, output_canvas, weight_canvas, processed_tile, weights, position):
        """Accumulate weighted tile into output canvas"""
        pos_x, pos_y = position
        tile_height, tile_width = processed_tile.shape[0], processed_tile.shape[1]

        # Move weights to same device
        weights = weights.to(output_canvas.device)

        # Accumulate weighted tile
        output_canvas[pos_y:pos_y + tile_height, pos_x:pos_x + tile_width] += processed_tile * weights.unsqueeze(-1)
        weight_canvas[pos_y:pos_y + tile_height, pos_x:pos_x + tile_width] += weights


# Node mappings
NODE_CLASS_MAPPINGS = {
    "RegionalTiledDiffusion": RegionalTiledDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionalTiledDiffusion": "Regional Tiled Diffusion",
}