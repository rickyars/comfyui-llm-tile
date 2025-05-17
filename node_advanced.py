import json
import torch
import comfy.sample
from comfy.utils import ProgressBar

from .utils import gaussian_blend_tiles, parse_tile_prompts
from .utils import combine_guider_conditioning, restore_guider_conditioning, apply_controlnet_to_guider_direct


class TiledImageGeneratorAdvanced:
    """
    ComfyUI node that generates a tiled image composition with overlapping regions
    using custom samplers and guiders.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_tile_prompts": ("STRING", {"multiline": True}),
                "grid_width": ("INT", {"default": 4, "min": 1, "max": 16}),
                "grid_height": ("INT", {"default": 6, "min": 1, "max": 16}),
                "tile_width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "tile_height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "overlap_percent": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.5, "step": 0.01}),
                "blend_sigma": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "controlnet": ("CONTROL_NET",),
                "controlnet_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("composite_image", "individual_tiles")
    FUNCTION = "generate_tiled_image"
    CATEGORY = "image/generation"

    def generate_tiled_image(self, json_tile_prompts, grid_width, grid_height,
                             tile_width, tile_height, overlap_percent, blend_sigma,
                             controlnet, controlnet_strength, seed, noise, guider,
                             sampler, sigmas, clip, vae):
        """Generate a tiled image composition with proper overlapping and seeding using advanced sampling."""

        # Parse the JSON tile prompts
        tile_prompts = parse_tile_prompts(json_tile_prompts, grid_width, grid_height)

        # Calculate overlap in pixels
        overlap_x = int(tile_width * overlap_percent)
        overlap_y = int(tile_height * overlap_percent)

        # Calculate final image dimensions
        final_width = tile_width + (grid_width - 1) * (tile_width - overlap_x)
        final_height = tile_height + (grid_height - 1) * (tile_height - overlap_y)

        # Create a blank canvas for the final composite
        final_tensor = torch.ones((1, final_height, final_width, 3), dtype=torch.float32) * 0.5

        # Storage for individual tiles
        individual_tensors = []
        positions = []

        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Calculate total number of tiles
        total_tiles = grid_width * grid_height
        pbar = ProgressBar(total_tiles)

        # Determine the guider type for proper handling
        guider_type = guider.__class__.__name__ if hasattr(guider, "__class__") else "Unknown"
        print(f"Using guider of type: {guider_type}")

        # Generate tiles one by one
        for y in range(grid_height):
            for x in range(grid_width):
                idx = y * grid_width + x
                current_prompt = tile_prompts[idx]
                current_seed = seed + idx

                print(f"Generating tile ({x + 1},{y + 1}) with prompt: {current_prompt}")

                # Calculate position in the final image
                pos_x = x * (tile_width - overlap_x)
                pos_y = y * (tile_height - overlap_y)

                # Use the original guider (no cloning)
                tile_guider = guider

                # Combine tile prompt with guider's global conditioning
                original_conditioning = combine_guider_conditioning(tile_guider, current_prompt, clip)

                # For first tile
                if x == 0 and y == 0:
                    # Standard sampling without mask
                    latent_height = tile_height // 8
                    latent_width = tile_width // 8

                    # Create an empty latent image
                    latent_image = torch.zeros([1, 4, latent_height, latent_width], device=device)

                    # Generate the tile noise
                    if noise is not None:
                        # Use provided noise
                        tile_noise = noise.generate_noise({"samples": latent_image})
                    else:
                        # Create new noise
                        tile_noise = comfy.sample.prepare_noise(latent_image, current_seed)

                    # Sample using the guider
                    samples = tile_guider.sample(
                        tile_noise,
                        latent_image,
                        sampler,
                        sigmas,
                        denoise_mask=None,
                        disable_pbar=False,
                        seed=current_seed
                    )

                    # Decode
                    tile_tensor = vae.decode(samples)

                else:
                    # For subsequent tiles, use context from previous tiles
                    working_tensor = torch.ones((1, tile_height, tile_width, 3), dtype=torch.float32) * 0.5
                    outpaint_mask = torch.ones((1, tile_height, tile_width, 1), dtype=torch.float32)

                    # Copy the overlapping regions from previous tiles
                    if x > 0:  # Left overlap
                        left_overlap_width = overlap_x
                        working_tensor[0, :, :left_overlap_width, :] = final_tensor[
                                                                       0,
                                                                       pos_y:pos_y + tile_height,
                                                                       pos_x:pos_x + left_overlap_width,
                                                                       :
                                                                       ]
                        outpaint_mask[0, :, :left_overlap_width, :] = 0

                    if y > 0:  # Top overlap
                        top_overlap_height = overlap_y
                        working_tensor[0, :top_overlap_height, :, :] = final_tensor[
                                                                       0,
                                                                       pos_y:pos_y + top_overlap_height,
                                                                       pos_x:pos_x + tile_width,
                                                                       :
                                                                       ]
                        outpaint_mask[0, :top_overlap_height, :, :] = 0

                    # Apply controlnet to guider if available
                    if controlnet is not None and controlnet_strength > 0:
                        apply_controlnet_to_guider_direct(tile_guider, controlnet, working_tensor, controlnet_strength,
                                                          vae)

                    # Convert working tensor to latent space
                    with torch.no_grad():
                        latent_image = vae.encode(working_tensor)

                    # Generate the tile noise
                    if noise is not None:
                        # Use provided noise
                        tile_noise = noise.generate_noise({"samples": latent_image})
                    else:
                        # Create new noise
                        tile_noise = comfy.sample.prepare_noise(latent_image, current_seed)

                    # Sample using the guider - we're not using the mask here, relying on controlnet
                    samples = tile_guider.sample(
                        tile_noise,
                        latent_image,
                        sampler,
                        sigmas,
                        denoise_mask=None,  # Not using explicit mask, relying on controlnet
                        disable_pbar=False,
                        seed=current_seed
                    )

                    # Decode
                    tile_tensor = vae.decode(samples)

                # Store this tile
                individual_tensors.append(tile_tensor.clone())
                positions.append((pos_x, pos_y))

                # Place the tile in the final composite
                h = min(tile_height, final_height - pos_y)
                w = min(tile_width, final_width - pos_x)
                final_tensor[0, pos_y:pos_y + h, pos_x:pos_x + w, :] = tile_tensor[0, :h, :w, :]

                # Restore original conditioning to guider
                restore_guider_conditioning(tile_guider, original_conditioning)

                # Update progress
                pbar.update(1)

        # Combine the individual tile tensors into a batch with Gaussian blending
        if individual_tensors:
            final_tensor = gaussian_blend_tiles(
                individual_tensors,
                positions,
                tile_width,
                tile_height,
                overlap_x,
                overlap_y,
                final_width,
                final_height,
                sigma=blend_sigma
            )
            tile_batch = torch.cat(individual_tensors, dim=0)
        else:
            # Fallback if no tiles were created
            final_tensor = torch.zeros((1, final_height, final_width, 3), dtype=torch.float32)
            tile_batch = torch.zeros((1, tile_height, tile_width, 3), dtype=torch.float32, device=device)

        return final_tensor, tile_batch


# This part is needed for ComfyUI to recognize the nodes
NODE_CLASS_MAPPINGS = {
    "TiledImageGeneratorAdvanced": TiledImageGeneratorAdvanced,
}

# Add descriptions for the web UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageGeneratorAdvanced": "Tiled Image Generator Advanced",
}