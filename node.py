import torch
import comfy.sample
import comfy.controlnet
from comfy.utils import ProgressBar

from .utils import parse_tile_prompts
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
                "seamlessX": ("BOOLEAN", {"default": True, "tooltip": "If true, side of image will be seamless."}),
                "seamlessY": ("BOOLEAN", {"default": False, "tooltip": "If true, top/bottom of image will be seamless."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("composite_image", "individual_tiles")
    FUNCTION = "generate_tiled_image"
    CATEGORY = "image/generation"

    def generate_tiled_image(self, json_tile_prompts, global_positive, global_negative,
                             grid_width, grid_height, tile_width, tile_height,
                             overlap_percent, controlnet, controlnet_strength,
                             seed, model, clip, vae, sampler_name, scheduler, steps, cfg, seamlessX, seamlessY):
        """Generate a tiled image with variable generation canvas sizes and simple placement."""

        # Parse the JSON tile prompts
        tile_prompts = parse_tile_prompts(json_tile_prompts, grid_width, grid_height)

        # Calculate overlap in pixels
        overlap_x = int(tile_width * overlap_percent)
        overlap_y = int(tile_height * overlap_percent)

        # Final size stays the same
        final_width = tile_width * grid_width
        final_height = tile_height * grid_height

        print(f"Final output size: {final_width} x {final_height}")
        print(f"Tile size: {tile_width} x {tile_height}")
        print(f"Grid: {grid_width} x {grid_height}")
        print(f"Overlap: {overlap_x} x {overlap_y} pixels ({overlap_percent * 100}%)")

        # Define max generation canvas size for consistent storage
        max_gen_width = ((tile_width + overlap_x + 7) // 8) * 8
        max_gen_height = ((tile_height + overlap_y + 7) // 8) * 8

        # Create a blank canvas for the final composite
        final_tensor = torch.zeros((1, final_height, final_width, 3), dtype=torch.float32)

        # Storage for debugging (padded to consistent size)
        full_grown_tensors = []

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

                # Calculate generation canvas size based on tile position
                if x == 0 and y == 0:  # Top-left corner - no expansion needed
                    gen_width = ((tile_width + 7) // 8) * 8
                    gen_height = ((tile_height + 7) // 8) * 8
                elif x == 0:  # Left column - only expand vertically for top neighbor
                    gen_width = ((tile_width + 7) // 8) * 8
                    gen_height = ((tile_height + overlap_y + 7) // 8) * 8
                elif y == 0:  # Top row - only expand horizontally for left neighbor
                    gen_width = ((tile_width + overlap_x + 7) // 8) * 8
                    gen_height = ((tile_height + 7) // 8) * 8
                else:  # Interior tiles - expand both directions
                    gen_width = ((tile_width + overlap_x + 7) // 8) * 8
                    gen_height = ((tile_height + overlap_y + 7) // 8) * 8

                print(f"Tile ({x + 1},{y + 1}) generation canvas: {gen_width} x {gen_height}")

                # Create the combined prompt
                combined_prompt = f"{current_prompt} {global_positive}" if global_positive else current_prompt
                print(f"Generating tile ({x + 1},{y + 1}) with prompt: {current_prompt}")

                # Encode the combined prompt
                pos_tokens = clip.tokenize(combined_prompt)
                pos_cond = clip.encode_from_tokens_scheduled(pos_tokens)

                # Encode the negative prompt
                neg_tokens = clip.tokenize(global_negative)
                neg_cond = clip.encode_from_tokens_scheduled(neg_tokens)

                # Create working canvas with variable size
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

                        # Copy to working tensor left edge (after top overlap region if exists)
                        target_start_y = overlap_y if has_top_neighbor else 0
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

                # --- SEAMLESS WRAP: For last tile in row, also overlap from first tile in row ---
                if seamlessX and x == grid_width - 1 and overlap_x > 0:
                    # Wrap to first tile in this row
                    wrap_source_x = 0
                    wrap_target_x = gen_width - overlap_x
                    source_start_y = final_pos_y
                    source_end_y = min(final_pos_y + tile_height, final_height)
                    source_height = source_end_y - source_start_y

                    target_start_y = overlap_y if has_top_neighbor else 0
                    copy_height = min(source_height, gen_height - target_start_y)

                    if copy_height > 0 and wrap_target_x >= 0:
                        working_tensor[0,
                        target_start_y:target_start_y + copy_height,
                        wrap_target_x:wrap_target_x + overlap_x,
                        :] = final_tensor[0,
                             source_start_y:source_start_y + copy_height,
                             wrap_source_x:wrap_source_x + overlap_x,
                             :]
                        outpaint_mask[0,
                        target_start_y:target_start_y + copy_height,
                        wrap_target_x:wrap_target_x + overlap_x,
                        :] = 0

                # --- SEAMLESS WRAP: For last tile in column, also overlap from first tile in column ---
                if seamlessY and y == grid_height - 1 and overlap_y > 0:
                    # Wrap to first tile in this column
                    wrap_source_y = 0
                    wrap_target_y = gen_height - overlap_y
                    source_start_x = final_pos_x
                    source_end_x = min(final_pos_x + tile_width, final_width)
                    source_width = source_end_x - source_start_x

                    target_start_x = overlap_x if has_left_neighbor else 0
                    copy_width = min(source_width, gen_width - target_start_x)

                    if copy_width > 0 and wrap_target_y >= 0:
                        working_tensor[0,
                        wrap_target_y:wrap_target_y + overlap_y,
                        target_start_x:target_start_x + copy_width,
                        :] = final_tensor[0,
                             wrap_source_y:wrap_source_y + overlap_y,
                             source_start_x:source_start_x + copy_width,
                             :]
                        outpaint_mask[0,
                        wrap_target_y:wrap_target_y + overlap_y,
                        target_start_x:target_start_x + copy_width,
                        :] = 0

                if has_top_neighbor:
                    # Copy top overlap region from previous tile
                    source_y = final_pos_y - overlap_y
                    if source_y >= 0 and source_y + overlap_y <= final_height:
                        source_start_x = final_pos_x
                        source_end_x = min(final_pos_x + tile_width, final_width)
                        source_width = source_end_x - source_start_x

                        # Copy to working tensor top edge (after left overlap region if exists)
                        target_start_x = overlap_x if has_left_neighbor else 0
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

                # Get latent shape for the variable canvas
                with torch.no_grad():
                    latent_shape = vae.encode(working_tensor).shape

                # Create empty latent tensor with actual generation shape
                latent_image = torch.zeros(latent_shape, device=device)

                # Apply ControlNet to conditioning using the variable canvas
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

                # Generate noise for variable canvas size
                noise = comfy.sample.prepare_noise(latent_image, current_seed, None)

                # Sample with the variable latent
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

                # Decode the variable canvas
                original_tile = vae.decode(samples)[0]  # Remove batch dimension

                # (1) CREATE PADDED VERSION FOR DEBUG OUTPUT
                padded_tensor = torch.zeros((max_gen_height, max_gen_width, 3),
                                            dtype=original_tile.dtype, device=original_tile.device)
                actual_h, actual_w = original_tile.shape[0], original_tile.shape[1]

                # Position content consistently for debug viewing
                if x == 0 and y == 0:
                    padded_tensor[overlap_y:overlap_y + actual_h, overlap_x:overlap_x + actual_w, :] = original_tile
                elif x == 0:
                    padded_tensor[0:actual_h, overlap_x:overlap_x + actual_w, :] = original_tile
                elif y == 0:
                    padded_tensor[overlap_y:overlap_y + actual_h, 0:actual_w, :] = original_tile
                else:
                    padded_tensor[0:actual_h, 0:actual_w, :] = original_tile

                full_grown_tensors.append(padded_tensor.unsqueeze(0))

                # (2) EXTRACT THE RIGHT 1024x1024 PORTION (NEW CONTENT)
                if x == 0 and y == 0:
                    # First tile: use entire content (1024x1024)
                    extracted_tile = original_tile[:tile_height, :tile_width, :]

                elif x == 0:  # Left column
                    # Skip top overlap (seeded), use bottom 1024x1024 (new content)
                    start_y = original_tile.shape[0] - tile_height  # Bottom portion
                    extracted_tile = original_tile[start_y:start_y + tile_height, :tile_width, :]

                elif y == 0:  # Top row
                    # Skip left overlap (seeded), use right 1024x1024 (new content)
                    start_x = original_tile.shape[1] - tile_width  # Right portion
                    extracted_tile = original_tile[:tile_height, start_x:start_x + tile_width, :]

                else:  # Interior tiles
                    # Skip both overlaps (seeded), use bottom-right 1024x1024 (new content)
                    start_y = original_tile.shape[0] - tile_height  # Bottom portion
                    start_x = original_tile.shape[1] - tile_width  # Right portion
                    extracted_tile = original_tile[start_y:start_y + tile_height, start_x:start_x + tile_width, :]

                # (3) PLACE AT GRID POSITION (NO BLENDING FOR NOW)
                final_tensor[0, final_pos_y:final_pos_y + tile_height, final_pos_x:final_pos_x + tile_width,
                :] = extracted_tile

                # Update progress
                pbar.update(1)

        # Return final tensor and debug tiles
        if full_grown_tensors:
            tile_batch = torch.cat(full_grown_tensors, dim=0)
        else:
            tile_batch = torch.zeros((1, max_gen_height, max_gen_width, 3), dtype=torch.float32, device=device)

        # Assume final_tensor is [1, H, W, C]
        if seamlessX and overlap_x > 0:
            trimmed_width = final_width - overlap_x
            final_tensor = final_tensor[:, :, :trimmed_width, :]
        if seamlessY and overlap_y > 0:
            trimmed_height = final_height - overlap_y
            final_tensor = final_tensor[:, :trimmed_height, :, :]

        return final_tensor, tile_batch


# This part is needed for ComfyUI to recognize the nodes
NODE_CLASS_MAPPINGS = {
    "TiledImageGenerator": TiledImageGenerator,
}

# Add descriptions for the web UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageGenerator": "Tiled Image Generator",
}