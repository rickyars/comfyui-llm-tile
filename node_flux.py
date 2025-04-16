import json
import torch


class TiledFluxGenerator:
    """
    ComfyUI node that generates a tiled image composition based on LLM-generated prompts
    using standard inpainting instead of ControlNet for creating seamless multi-tile compositions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "json_tile_prompts": ("STRING", {"multiline": True}),
                "global_positive": ("STRING", {"multiline": True, "default": ""}),
                "global_negative": ("STRING", {"multiline": True, "default": ""}),
                "grid_width": ("INT", {"default": 4, "min": 1, "max": 8}),
                "grid_height": ("INT", {"default": 6, "min": 1, "max": 8}),
                "tile_width": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "tile_height": ("INT", {"default": 512, "min": 256, "max": 1024}),
                "overlap_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "feathering": ("INT", {"default": 40, "min": 0, "max": 512, "step": 1}),
                "flux_guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
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

    def _conditioning_set_values(self, conditioning, values):
        """Helper method to set values in conditioning."""
        c = []
        for t in conditioning:
            d = t[1].copy()
            for k, v in values.items():
                d[k] = v
            n = [t[0], d]
            c.append(n)
        return c

    def generate_tiled_image(self, json_tile_prompts, global_positive, global_negative,
                             grid_width, grid_height, tile_width, tile_height,
                             overlap_percent, seed, model, clip, vae, feathering,
                             flux_guidance, steps, cfg, sampler_name, scheduler):
        """Generate a tiled image composition using inpainting."""
        # Import directly from ComfyUI
        from nodes import common_ksampler

        # Parse the JSON tile prompts
        tile_prompts = self._parse_tile_prompts(json_tile_prompts, grid_width, grid_height)

        # Calculate overlap in pixels
        overlap_x = int(tile_width * overlap_percent)
        overlap_y = int(tile_height * overlap_percent)

        # Calculate final image dimensions
        final_width = tile_width + (grid_width - 1) * (tile_width - overlap_x)
        final_height = tile_height + (grid_height - 1) * (tile_height - overlap_y)

        # Create a blank canvas for the final composite (gray fill)
        final_tensor = torch.ones((1, final_height, final_width, 3), dtype=torch.float32) * 0.5

        # Storage for individual tiles
        individual_tensors = []

        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Generate tiles one by one
        for y in range(grid_height):
            for x in range(grid_width):
                idx = y * grid_width + x
                current_prompt = tile_prompts[idx]
                current_seed = seed + idx

                # Create the combined prompt
                combined_prompt = f"{current_prompt} {global_positive}" if global_positive else current_prompt
                print(f"Generating tile ({x + 1},{y + 1}) with prompt: {combined_prompt}")

                # Encode the combined prompt
                pos_tokens = clip.tokenize(combined_prompt)
                pos_cond = clip.encode_from_tokens_scheduled(pos_tokens)

                # Encode the negative prompt
                neg_tokens = clip.tokenize(global_negative)
                neg_cond = clip.encode_from_tokens_scheduled(neg_tokens)

                # Flux Guidance
                if flux_guidance is not None and flux_guidance > 0:
                    pos_cond = self._conditioning_set_values(pos_cond, {"guidance": flux_guidance})
                    neg_cond = self._conditioning_set_values(neg_cond, {"guidance": flux_guidance})

                # Calculate position in the final image
                pos_x = x * (tile_width - overlap_x)
                pos_y = y * (tile_height - overlap_y)

                # For first tile, simply generate
                if x == 0 and y == 0:
                    # Standard sampling without inpainting
                    latent_height = tile_height // 8
                    latent_width = tile_width // 8

                    # Create an empty latent image DICTIONARY as required by common_ksampler()
                    latent_tensor = torch.zeros([1, 4, latent_height, latent_width], device=device)
                    latent_dict = {"samples": latent_tensor}

                    # Sample using common_ksampler
                    output = common_ksampler(
                        model,
                        current_seed,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        pos_cond,
                        neg_cond,
                        latent_dict,
                        denoise=1.0
                    )

                    latent_output = output[0]  # common_ksampler returns a tuple

                    # Decode
                    tile_tensor = vae.decode(latent_output["samples"])

                else:
                    # For subsequent tiles, use inpainting with context from already generated tiles

                    # Create a completely gray canvas for this tile (instead of black)
                    working_tensor = torch.ones((1, tile_height, tile_width, 3), dtype=torch.float32) * 0.5

                    # Create a mask where 1 = areas to generate, 0 = keep black areas
                    outpaint_mask = torch.ones((1, tile_height, tile_width, 1), dtype=torch.float32)

                    # After setting the overlap regions in the mask to 0
                    if feathering > 0:
                        # Apply feathering at the left edge if needed
                        if x > 0:
                            for j in range(min(feathering, tile_width - overlap_x)):
                                pos = overlap_x + j
                                factor = j / feathering
                                outpaint_mask[0, :, pos, :] = factor * factor

                        # Apply feathering at the top edge if needed
                        if y > 0:
                            for i in range(min(feathering, tile_height - overlap_y)):
                                pos = overlap_y + i
                                factor = i / feathering
                                outpaint_mask[0, i, :, :] = factor * factor

                    if x > 0 or y > 0:  # If this isn't the first tile
                        # Copy the overlapping regions from previous tiles
                        if x > 0:  # Left overlap
                            left_overlap_width = overlap_x
                            # Copy from the right edge of the previous tile
                            working_tensor[0, :, :left_overlap_width, :] = final_tensor[
                                                                           0,
                                                                           pos_y:pos_y + tile_height,
                                                                           pos_x:pos_x + left_overlap_width,
                                                                           :
                                                                           ]
                            # Mark these areas as "keep" in the mask
                            outpaint_mask[0, :, :left_overlap_width, :] = 0

                        if y > 0:  # Top overlap
                            top_overlap_height = overlap_y
                            # Copy from the bottom edge of the tile above
                            working_tensor[0, :top_overlap_height, :, :] = final_tensor[
                                                                           0,
                                                                           pos_y:pos_y + top_overlap_height,
                                                                           pos_x:pos_x + tile_width,
                                                                           :
                                                                           ]
                            # Mark these areas as "keep" in the mask
                            outpaint_mask[0, :top_overlap_height, :, :] = 0

                    # Get a properly sized empty latent shape
                    with torch.no_grad():
                        latent_image = vae.encode(working_tensor)

                    # Create a latent dictionary for common_ksampler
                    latent_dict = {"samples": latent_image}

                    # Properly reshape the mask for the VAE before setting it in the conditioning
                    vae_mask = outpaint_mask.permute(0, 3, 1, 2)

                    # Modify conditioning to include latent and mask
                    inpaint_positive = self._conditioning_set_values(pos_cond, {
                        "concat_latent_image": latent_image,
                        "concat_mask": vae_mask,
                        "guidance": flux_guidance  # Add guidance here again
                    })

                    inpaint_negative = self._conditioning_set_values(neg_cond, {
                        "concat_latent_image": latent_image,
                        "concat_mask": vae_mask,
                        "guidance": flux_guidance  # Add guidance here again
                    })

                    # Sample using common_ksampler for inpainting
                    output = common_ksampler(
                        model,
                        current_seed,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        inpaint_positive,
                        inpaint_negative,
                        latent_dict,
                        denoise=1.0
                    )

                    latent_output = output[0]  # common_ksampler returns a tuple

                    # Decode
                    tile_tensor = vae.decode(latent_output["samples"])

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

        return final_tensor, tile_batch

    def _parse_tile_prompts(self, json_string, grid_width, grid_height):
        """Parse the JSON tile prompts with more flexible position handling."""
        try:
            data = json.loads(json_string)

            # Validate the JSON structure
            if not isinstance(data, list):
                raise ValueError("JSON must contain a list of tile objects")

            expected_tiles = grid_width * grid_height
            if len(data) != expected_tiles:
                print(
                    f"Warning: Expected {expected_tiles} tile prompts, got {len(data)}. Proceeding with available data.")

            # Extract the prompts in the correct order
            tile_prompts = []
            for y in range(grid_height):
                for x in range(grid_width):
                    idx = y * grid_width + x
                    if idx < len(data):
                        tile_info = data[idx]

                        # Check for required fields
                        if "prompt" not in tile_info:
                            raise ValueError(f"Tile at index {idx} is missing required 'prompt' field")

                        # If position is missing or incorrect, print a warning but proceed
                        if "position" not in tile_info:
                            print(
                                f"Warning: Tile at index {idx} is missing 'position' field. Using grid position ({x + 1},{y + 1}).")
                        else:
                            pos = tile_info["position"]
                            expected_x = x + 1
                            expected_y = y + 1
                            if pos.get("x") != expected_x or pos.get("y") != expected_y:
                                print(
                                    f"Warning: Tile at index {idx} has position {pos} but expected ({expected_x},{expected_y}). Using prompt anyway.")

                        tile_prompts.append(tile_info["prompt"])
                    else:
                        # If we're missing data, use a default prompt
                        default_prompt = f"Generate content for tile at position ({x + 1},{y + 1})"
                        print(f"Warning: Missing data for tile at index {idx}. Using default prompt.")
                        tile_prompts.append(default_prompt)

            return tile_prompts

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error parsing JSON: {str(e)}")
