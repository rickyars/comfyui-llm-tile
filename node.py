import torch
import comfy.sample
import comfy.controlnet
from comfy.utils import ProgressBar

from .utils import gaussian_blend_tiles, parse_tile_prompts
from .utils import apply_controlnet_to_conditioning

class TiledImageGenerator:
    """
    ComfyUI node that generates a tiled image composition with overlapping regions
    that serve as seeds for neighboring tiles, using ControlNet Union SDXL for outpainting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "json_tile_prompts": ("STRING", {"multiline": True}),
                "global_positive": ("STRING", {"multiline": True, "default": ""}),
                "global_negative": ("STRING", {"multiline": True, "default": ""}),
                "grid_width": ("INT", {"default": 4, "min": 1, "max": 16}),
                "grid_height": ("INT", {"default": 6, "min": 1, "max": 16}),
                "tile_width": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "tile_height": ("INT", {"default": 1024, "min": 256, "max": 2048}),
                "overlap_percent": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.5, "step": 0.01}),
                "blend_sigma": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "controlnet": ("CONTROL_NET",),
                "controlnet_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("composite_image", "individual_tiles")
    FUNCTION = "generate_tiled_image"
    CATEGORY = "image/generation"

    def generate_tiled_image(self, json_tile_prompts, global_positive, global_negative,
                             grid_width, grid_height, tile_width, tile_height,
                             overlap_percent, blend_sigma, controlnet, controlnet_strength,
                             seed, model, clip, vae, sampler_name, scheduler, steps, cfg):
        """Generate a tiled image with uniform grown latents for all tiles."""

        # Parse the JSON tile prompts
        tile_prompts = parse_tile_prompts(json_tile_prompts, grid_width, grid_height)

        # Calculate overlap in pixels
        overlap_x = int(tile_width * overlap_percent)
        overlap_y = int(tile_height * overlap_percent)

        # SIMPLE FINAL SIZE CALCULATION - exact grid multiplication
        final_width = tile_width * grid_width
        final_height = tile_height * grid_height

        print(f"Final output size: {final_width} x {final_height}")
        print(f"Tile size: {tile_width} x {tile_height}")
        print(f"Grid: {grid_width} x {grid_height}")
        print(f"Overlap: {overlap_x} x {overlap_y} pixels ({overlap_percent * 100}%)")

        # UNIFORM GENERATION CANVAS SIZE for all tiles
        gen_width = ((tile_width + overlap_x + 7) // 8) * 8
        gen_height = ((tile_height + overlap_y + 7) // 8) * 8
        print(f"Uniform generation canvas: {gen_width} x {gen_height}")

        # Create a blank canvas for the final composite
        final_tensor = torch.ones((1, final_height, final_width, 3), dtype=torch.float32) * 0.5

        # Storage for individual tiles and full grown tiles
        individual_tensors = []  # For blending (cropped)
        full_grown_tensors = []  # For debugging (full canvas)
        positions = []

        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Calculate total number of tiles
        total_tiles = grid_width * grid_height
        pbar = ProgressBar(total_tiles)

        # Generate tiles one by one
        for y in range(grid_height):
            for x in range(grid_width):
                idx = y * grid_width + x
                current_prompt = tile_prompts[idx]
                current_seed = seed + idx

                # Calculate position in the final image
                final_pos_x = x * tile_width
                final_pos_y = y * tile_height

                # Create the combined prompt
                combined_prompt = f"{current_prompt} {global_positive}" if global_positive else current_prompt
                print(f"Generating tile ({x + 1},{y + 1}) with prompt: {current_prompt}")

                # Encode the combined prompt
                pos_tokens = clip.tokenize(combined_prompt)
                pos_cond = clip.encode_from_tokens_scheduled(pos_tokens)

                # Encode the negative prompt
                neg_tokens = clip.tokenize(global_negative)
                neg_cond = clip.encode_from_tokens_scheduled(neg_tokens)

                # Create UNIFORM working canvas for ALL tiles
                working_tensor = torch.zeros((1, gen_height, gen_width, 3), dtype=torch.float32)
                outpaint_mask = torch.ones((1, gen_height, gen_width, 1), dtype=torch.float32)

                # Check if we have neighbors to copy from
                has_left_neighbor = x > 0
                has_top_neighbor = y > 0

                # COPY OVERLAP REGIONS from final_tensor (only if neighbors exist)
                if has_left_neighbor:
                    # Copy left overlap region from previous tile
                    source_x = final_pos_x - overlap_x
                    if source_x >= 0 and source_x + overlap_x <= final_width:
                        source_start_y = final_pos_y
                        source_end_y = min(final_pos_y + tile_height, final_height)
                        source_height = source_end_y - source_start_y

                        # Copy to working tensor left edge (after top overlap region)
                        target_start_y = overlap_y
                        copy_height = min(source_height, gen_height - target_start_y)

                        if copy_height > 0:
                            working_tensor[0,
                            target_start_y:target_start_y + copy_height,
                            :overlap_x,
                            :] = final_tensor[0,
                                 source_start_y:source_start_y + copy_height,
                                 source_x:source_x + overlap_x,
                                 :]
                            # Mark as "keep" in mask
                            outpaint_mask[0,
                            target_start_y:target_start_y + copy_height,
                            :overlap_x,
                            :] = 0

                if has_top_neighbor:
                    # Copy top overlap region from previous tile
                    source_y = final_pos_y - overlap_y
                    if source_y >= 0 and source_y + overlap_y <= final_height:
                        source_start_x = final_pos_x
                        source_end_x = min(final_pos_x + tile_width, final_width)
                        source_width = source_end_x - source_start_x

                        # Copy to working tensor top edge (after left overlap region)
                        target_start_x = overlap_x
                        copy_width = min(source_width, gen_width - target_start_x)

                        if copy_width > 0:
                            working_tensor[0,
                            :overlap_y,
                            target_start_x:target_start_x + copy_width,
                            :] = final_tensor[0,
                                 source_y:source_y + overlap_y,
                                 source_start_x:source_start_x + copy_width,
                                 :]
                            # Mark as "keep" in mask
                            outpaint_mask[0,
                            :overlap_y,
                            target_start_x:target_start_x + copy_width,
                            :] = 0

                # Handle corner overlap if both neighbors exist
                if has_left_neighbor and has_top_neighbor:
                    corner_source_x = final_pos_x - overlap_x
                    corner_source_y = final_pos_y - overlap_y
                    if (corner_source_x >= 0 and corner_source_y >= 0 and
                            corner_source_x + overlap_x <= final_width and
                            corner_source_y + overlap_y <= final_height):
                        working_tensor[0, :overlap_y, :overlap_x, :] = final_tensor[0,
                                                                       corner_source_y:corner_source_y + overlap_y,
                                                                       corner_source_x:corner_source_x + overlap_x,
                                                                       :]
                        outpaint_mask[0, :overlap_y, :overlap_x, :] = 0

                # Get latent shape for the uniform grown canvas
                with torch.no_grad():
                    latent_shape = vae.encode(working_tensor).shape

                # Create empty latent tensor with grown shape
                latent_image = torch.zeros(latent_shape, device=device)

                # Apply ControlNet to conditioning using the grown canvas
                conditioning = apply_controlnet_to_conditioning(
                    positive=pos_cond,
                    negative=neg_cond,
                    control_net=controlnet,
                    image=working_tensor,
                    strength=controlnet_strength,
                    start_percent=0.0,
                    end_percent=1.0,
                    vae=vae
                )

                # Generate noise for grown canvas size
                noise = comfy.sample.prepare_noise(latent_image, current_seed, None)

                # Sample with the grown latent
                samples = comfy.sample.sample(
                    model,
                    noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    conditioning[0],
                    conditioning[1],
                    latent_image
                )

                # Decode the full grown canvas
                full_grown_tensor = vae.decode(samples)

                # EXTRACT the tile_width x tile_height region from consistent position
                # Always extract from overlap_x, overlap_y for uniform behavior
                extract_x = overlap_x
                extract_y = overlap_y

                extracted_tile = full_grown_tensor[0:1,
                                 extract_y:extract_y + tile_height,
                                 extract_x:extract_x + tile_width,
                                 :]

                # Store both versions
                individual_tensors.append(extracted_tile.clone())  # For blending
                full_grown_tensors.append(full_grown_tensor.clone())  # For debugging
                positions.append((final_pos_x, final_pos_y))

                # Place the tile in the final composite at exact grid position
                final_tensor[0,
                final_pos_y:final_pos_y + tile_height,
                final_pos_x:final_pos_x + tile_width,
                :] = extracted_tile[0, :, :, :]

                # Update progress
                pbar.update(1)

        # Apply Gaussian blending for smooth transitions
        if individual_tensors:
            final_tensor = gaussian_blend_tiles(
                individual_tensors,  # Use cropped tiles for blending
                positions,
                tile_width,
                tile_height,
                overlap_x,
                overlap_y,
                final_width,
                final_height,
                sigma=blend_sigma
            )
            # Return full grown tiles for debugging
            tile_batch = torch.cat(full_grown_tensors, dim=0)
        else:
            final_tensor = torch.zeros((1, final_height, final_width, 3), dtype=torch.float32)
            tile_batch = torch.zeros((1, gen_height, gen_width, 3), dtype=torch.float32, device=device)

        return final_tensor, tile_batch

# This part is needed for ComfyUI to recognize the nodes
NODE_CLASS_MAPPINGS = {
    "TiledImageGenerator": TiledImageGenerator,
}

# Add descriptions for the web UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageGenerator": "Tiled Image Generator",
}
