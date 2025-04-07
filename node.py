import json
import os
import numpy as np
from PIL import Image, ImageDraw
import torch


class TiledImageGenerator:
    """
    ComfyUI node that generates a tiled image composition with overlapping regions
    that serve as seeds for neighboring tiles.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_tile_prompts": ("STRING", {"multiline": True}),
                "grid_width": ("INT", {"default": 4, "min": 1, "max": 8}),
                "grid_height": ("INT", {"default": 6, "min": 1, "max": 8}),
                "tile_width": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "tile_height": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "overlap_percent": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.5}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": 100}),
                "base_seed": ("INT", {"default": 0}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
                                  "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde",
                                  "dpmpp_3m_sde", "ddim", "uni_pc", "uni_pc_bh2"],),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("composite_image", "individual_tiles")
    FUNCTION = "generate_tiled_image"
    CATEGORY = "image/generation"

    def generate_tiled_image(self, json_tile_prompts, grid_width, grid_height,
                             tile_width, tile_height, overlap_percent, feathering, base_seed,
                             model, clip, vae, positive, negative, steps, cfg, sampler_name, scheduler):
        """Generate a tiled image composition with proper overlapping and seeding."""

        try:
            # Parse the JSON tile prompts
            tile_prompts = self._parse_tile_prompts(json_tile_prompts, grid_width, grid_height)

            # Calculate overlap in pixels
            overlap_x = int(tile_width * overlap_percent)
            overlap_y = int(tile_height * overlap_percent)

            # Calculate final image dimensions
            final_width = tile_width + (grid_width - 1) * (tile_width - overlap_x)
            final_height = tile_height + (grid_height - 1) * (tile_height - overlap_y)

            # Create a blank canvas for the final composite
            # [B, H, W, C] format for ComfyUI
            final_tensor = torch.ones((1, final_height, final_width, 3), dtype=torch.float32) * 0.5

            # Storage for individual tiles
            individual_tensors = []

            # Get device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Import ComfyUI's sampling functions
            import comfy.sample
            import comfy.samplers

            # Generate tiles one by one
            for y in range(grid_height):
                for x in range(grid_width):
                    idx = y * grid_width + x
                    current_prompt = tile_prompts[idx]
                    current_seed = base_seed + idx

                    print(f"Generating tile ({x + 1},{y + 1}) with prompt: {current_prompt}")

                    # Create tile-specific conditioning
                    tokens = clip.tokenize(current_prompt)
                    pos_cond = clip.encode_from_tokens_scheduled(tokens)

                    # Calculate position in the final image
                    pos_x = x * (tile_width - overlap_x)
                    pos_y = y * (tile_height - overlap_y)

                    # For first tile, simply generate
                    if x == 0 and y == 0:
                        # Standard sampling without mask
                        latent_height = tile_height // 8
                        latent_width = tile_width // 8

                        # Create noise
                        torch.manual_seed(current_seed)
                        noise = torch.randn([1, 4, latent_height, latent_width], device=device)

                        # Sample
                        # Create an empty latent image as required by sample()
                        latent_image = torch.zeros([1, 4, latent_height, latent_width], device=device)

                        samples = comfy.sample.sample(
                            model,
                            noise,
                            steps,
                            cfg,
                            sampler_name,
                            scheduler,
                            pos_cond,
                            negative,
                            latent_image=latent_image
                        )

                        # Decode
                        tile_tensor = vae.decode(samples)

                        # Check if permutation is needed
                        print(f"Shape after VAE decode: {tile_tensor.shape}")
                        if tile_tensor.shape[1] == 3 and tile_tensor.shape[3] != 3:
                            # If we have [B, C, H, W] format, permute to [B, H, W, C]
                            tile_tensor = tile_tensor.permute(0, 2, 3, 1)
                        else:
                            # If already in [B, H, W, C] format, no permutation needed
                            pass
                        print(f"Final tile tensor shape: {tile_tensor.shape}")
                    else:
                        # For subsequent tiles, use inpainting approach

                        # Create a working region for this tile
                        working_tensor = torch.ones((1, tile_height, tile_width, 3), dtype=torch.float32) * 0.5

                        # Create a mask (1 = generate new, 0 = keep original)
                        mask = torch.ones((1, tile_height, tile_width, 1), dtype=torch.float32)

                        # Copy existing content from the final image if there's overlap
                        has_overlap = False

                        # Handle corner overlap first (when both x > 0 and y > 0)
                        if x > 0 and y > 0:
                            # We have both left and top overlap
                            left_overlap_width = overlap_x
                            top_overlap_height = overlap_y

                            # Copy the corner region
                            working_tensor[0, :top_overlap_height, :left_overlap_width, :] = final_tensor[
                                                                                             0,
                                                                                             pos_y:pos_y + top_overlap_height,
                                                                                             pos_x:pos_x + left_overlap_width,
                                                                                             :
                                                                                             ]

                            # Mark corner as region to keep in mask
                            mask[0, :top_overlap_height, :left_overlap_width, :] = 0
                            has_overlap = True

                            # Copy left edge (excluding corner)
                            working_tensor[0, top_overlap_height:, :left_overlap_width, :] = final_tensor[
                                                                                             0,
                                                                                             pos_y + top_overlap_height:pos_y + tile_height,
                                                                                             pos_x:pos_x + left_overlap_width,
                                                                                             :
                                                                                             ]

                            # Mark left edge as region to keep
                            mask[0, top_overlap_height:, :left_overlap_width, :] = 0

                            # Copy top edge (excluding corner)
                            working_tensor[0, :top_overlap_height, left_overlap_width:, :] = final_tensor[
                                                                                             0,
                                                                                             pos_y:pos_y + top_overlap_height,
                                                                                             pos_x + left_overlap_width:pos_x + tile_width,
                                                                                             :
                                                                                             ]

                            # Mark top edge as region to keep
                            mask[0, :top_overlap_height, left_overlap_width:, :] = 0

                            # Apply feathering to both edges
                            self._apply_horizontal_feathering(mask, left_overlap_width, feathering, left=True)
                            self._apply_vertical_feathering(mask, top_overlap_height, feathering, top=True)

                        # Otherwise handle individual edge cases
                        elif x > 0:
                            # Normal left overlap handling (existing code)
                            left_overlap_width = overlap_x
                            src_x = pos_x

                            # Copy the right edge of the previous tile
                            working_tensor[0, :, :left_overlap_width, :] = final_tensor[
                                                                           0,
                                                                           pos_y:pos_y + tile_height,
                                                                           src_x:src_x + left_overlap_width,
                                                                           :
                                                                           ]

                            # Mark as region to keep in mask
                            mask[0, :, :left_overlap_width, :] = 0
                            has_overlap = True

                            # Apply feathering to mask
                            self._apply_horizontal_feathering(mask, left_overlap_width, feathering, left=True)

                        elif y > 0:
                            # Normal top overlap handling (existing code)
                            top_overlap_height = overlap_y
                            src_y = pos_y

                            # Copy the bottom edge of the tile above
                            working_tensor[0, :top_overlap_height, :, :] = final_tensor[
                                                                           0,
                                                                           src_y:src_y + top_overlap_height,
                                                                           pos_x:pos_x + tile_width,
                                                                           :
                                                                           ]

                            # Mark as region to keep in mask
                            mask[0, :top_overlap_height, :, :] = 0
                            has_overlap = True

                            # Apply feathering to mask
                            self._apply_vertical_feathering(mask, top_overlap_height, feathering, top=True)

                        # If we have overlap, use inpainting
                        if has_overlap or y > 0:
                            # Debug the working tensor shape
                            print(f"Working tensor shape before permute: {working_tensor.shape}")

                            # Encode to latent space - THIS LINE WAS MISSING
                            latent = vae.encode(working_tensor)
                            print(f"Latent shape after encoding: {latent.shape}")

                            # Prepare mask for sampling (needs [B, 1, H, W] format)
                            mask_vae = mask.permute(0, 3, 1, 2)

                            # Debug mask shape
                            print(f"Mask shape after permute: {mask_vae.shape}")

                            latent_mask = torch.nn.functional.interpolate(
                                mask_vae, size=(tile_height // 8, tile_width // 8), mode="bilinear"
                            )

                            # Generate noise for sampling
                            torch.manual_seed(current_seed)
                            noise = torch.randn([1, 4, tile_height // 8, tile_width // 8], device=device)

                            # Sample with inpainting
                            samples = comfy.sample.sample(
                                model,
                                noise,
                                steps,
                                cfg,
                                sampler_name,
                                scheduler,
                                pos_cond,
                                negative,
                                latent_image=latent,
                                noise_mask=latent_mask
                            )

                            # Decode
                            tile_tensor = vae.decode(samples)

                            # Check if permutation is needed
                            print(f"Shape after VAE decode: {tile_tensor.shape}")
                            if tile_tensor.shape[1] == 3 and tile_tensor.shape[3] != 3:
                                # If we have [B, C, H, W] format, permute to [B, H, W, C]
                                tile_tensor = tile_tensor.permute(0, 2, 3, 1)
                            else:
                                # If already in [B, H, W, C] format, no permutation needed
                                pass
                            print(f"Final tile tensor shape: {tile_tensor.shape}")
                        else:
                            # No overlap, just generate normally
                            latent_height = tile_height // 8
                            latent_width = tile_width // 8

                            # Create noise
                            torch.manual_seed(current_seed)
                            noise = torch.randn([1, 4, latent_height, latent_width], device=device)

                            # Sample
                            # Create an empty latent image as required by sample()
                            latent_image = torch.zeros([1, 4, latent_height, latent_width], device=device)

                            samples = comfy.sample.sample(
                                model,
                                noise,
                                steps,
                                cfg,
                                sampler_name,
                                scheduler,
                                pos_cond,
                                negative,
                                latent_image=latent_image
                            )

                            # Decode
                            tile_tensor = vae.decode(samples)

                            # Check if permutation is needed
                            print(f"Shape after VAE decode: {tile_tensor.shape}")
                            if tile_tensor.shape[1] == 3 and tile_tensor.shape[3] != 3:
                                # If we have [B, C, H, W] format, permute to [B, H, W, C]
                                tile_tensor = tile_tensor.permute(0, 2, 3, 1)
                            else:
                                # If already in [B, H, W, C] format, no permutation needed
                                pass
                            print(f"Final tile tensor shape: {tile_tensor.shape}")

                    # Store this tile
                    individual_tensors.append(tile_tensor.clone())

                    # Place the tile in the final composite
                    h = min(tile_height, final_height - pos_y)
                    w = min(tile_width, final_width - pos_x)

                    final_tensor[0, pos_y:pos_y + h, pos_x:pos_x + w, :] = tile_tensor[0, :h, :w, :]

            # Combine the individual tile tensors into a batch
            if individual_tensors:
                tile_batch = torch.cat(individual_tensors, dim=0)
            else:
                # Fallback if no tiles were created
                tile_batch = torch.zeros((1, tile_height, tile_width, 3), dtype=torch.float32, device=device)

            return (final_tensor, tile_batch)

        except Exception as e:
            import traceback
            print(f"Error in tiled image generation: {str(e)}")
            traceback.print_exc()

            # Create an error tensor in the format expected by ComfyUI
            error_tensor = torch.ones((1, 512, 512, 3), dtype=torch.float32) * 0.5
            error_batch = error_tensor.clone()
            return (error_tensor, error_batch)

    def _parse_tile_prompts(self, json_string, grid_width, grid_height):
        """Parse the JSON tile prompts."""
        try:
            data = json.loads(json_string)

            # Validate the JSON structure
            if not isinstance(data, list):
                raise ValueError("JSON must contain a list of tile objects")

            expected_tiles = grid_width * grid_height
            if len(data) != expected_tiles:
                raise ValueError(f"Expected {expected_tiles} tile prompts, got {len(data)}")

            # Extract the prompts in the correct order
            tile_prompts = []
            for y in range(grid_height):
                for x in range(grid_width):
                    idx = y * grid_width + x
                    if idx < len(data):
                        tile_info = data[idx]

                        # Check for required fields
                        if "position" not in tile_info or "prompt" not in tile_info:
                            raise ValueError(f"Tile at index {idx} is missing 'position' or 'prompt'")

                        pos = tile_info["position"]
                        if pos["x"] != x + 1 or pos["y"] != y + 1:
                            raise ValueError(f"Tile at index {idx} has incorrect position")

                        tile_prompts.append(tile_info["prompt"])

            return tile_prompts

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing JSON: {str(e)}")

    def _apply_horizontal_feathering(self, mask, edge_width, feather_size, left=True):
        """Apply horizontal feathering to mask edge for smooth transition."""
        if feather_size <= 0:
            return

        height, width = mask.shape[1], mask.shape[2]

        # Determine the feathering range
        if left:
            start = edge_width
            end = min(edge_width + feather_size, width)
        else:
            start = max(0, width - edge_width - feather_size)
            end = width - edge_width

        # Apply feathering
        for x in range(start, end):
            # Calculate alpha value (0 to 1)
            if left:
                t = (x - edge_width) / feather_size
            else:
                t = (end - x) / feather_size

            # Quadratic falloff for smoother transition
            alpha = t * t

            # Create a tensor with the same shape as the column we're modifying
            alpha_tensor = torch.ones_like(mask[0, :, x, 0]) * alpha

            # Use maximum to preserve existing stronger keep regions (values closer to 0)
            mask[0, :, x, 0] = torch.maximum(alpha_tensor, mask[0, :, x, 0])

    def _apply_vertical_feathering(self, mask, edge_height, feather_size, top=True):
        """Apply vertical feathering to mask edge for smooth transition."""
        if feather_size <= 0:
            return

        height, width = mask.shape[1], mask.shape[2]

        # Determine the feathering range
        if top:
            start = edge_height
            end = min(edge_height + feather_size, height)
        else:
            start = max(0, height - edge_height - feather_size)
            end = height - edge_height

        # Apply feathering
        for y in range(start, end):
            # Calculate alpha value (0 to 1)
            if top:
                t = (y - edge_height) / feather_size
            else:
                t = (end - y) / feather_size

            # Quadratic falloff for smoother transition
            alpha = t * t

            # Create a tensor with the same shape as the row we're modifying
            alpha_tensor = torch.ones_like(mask[0, y, :, 0]) * alpha

            # Use maximum to preserve existing stronger keep regions (values closer to 0)
            mask[0, y, :, 0] = torch.maximum(alpha_tensor, mask[0, y, :, 0])
class TilePromptTemplateNode:
    """
    Helper node to generate a template JSON for the TiledImageGenerator
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "main_prompt": ("STRING", {"multiline": True}),
                "grid_width": ("INT", {"default": 4, "min": 1, "max": 8}),
                "grid_height": ("INT", {"default": 6, "min": 1, "max": 8})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_template",)
    FUNCTION = "generate_template"
    CATEGORY = "utils"

    def generate_template(self, main_prompt, grid_width, grid_height):
        """
        Generate a template JSON structure for tile prompts
        """
        template = []

        for y in range(grid_height):
            for x in range(grid_width):
                # Create position-aware description
                if x == 0 and y == 0:
                    position_desc = "top-left corner"
                elif x == grid_width - 1 and y == 0:
                    position_desc = "top-right corner"
                elif x == 0 and y == grid_height - 1:
                    position_desc = "bottom-left corner"
                elif x == grid_width - 1 and y == grid_height - 1:
                    position_desc = "bottom-right corner"
                elif x == 0:
                    position_desc = "left edge"
                elif x == grid_width - 1:
                    position_desc = "right edge"
                elif y == 0:
                    position_desc = "top edge"
                elif y == grid_height - 1:
                    position_desc = "bottom edge"
                else:
                    position_desc = "middle section"

                template.append({
                    "position": {"x": x + 1, "y": y + 1},
                    "prompt": f"{main_prompt}, {position_desc} of the composition"
                })

        return json.dumps(template, indent=2)


# This part is needed for ComfyUI to recognize the nodes
NODE_CLASS_MAPPINGS = {
    "TiledImageGenerator": TiledImageGenerator,
    "TilePromptTemplateNode": TilePromptTemplateNode
}

# Add descriptions for the web UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageGenerator": "Tiled Image Generator",
    "TilePromptTemplateNode": "Tile Prompt Template"
}