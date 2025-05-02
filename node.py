import json
import torch
import comfy.sample
import comfy.controlnet
from comfy.utils import ProgressBar

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
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("composite_image", "individual_tiles")
    FUNCTION = "generate_tiled_image"
    CATEGORY = "image/generation"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, vae=None,
                         extra_concat=[]):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1, 1)
        out = []

        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                prev_cnet = d.get('control', None)

                # Apply the controlnet directly without copying
                # Create a temporary controlnet application that doesn't persist
                temp_cnet = control_net.set_cond_hint(control_hint, strength,
                                                      (start_percent, end_percent),
                                                      vae=vae, extra_concat=extra_concat)

                # Set previous controlnet but don't store in a dictionary
                if prev_cnet is not None:
                    temp_cnet.set_previous_controlnet(prev_cnet)

                d['control'] = temp_cnet
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)

        return out[0], out[1]

    def generate_tiled_image(self, json_tile_prompts, global_positive, global_negative, grid_width, grid_height,
                             tile_width, tile_height, overlap_percent, seed,
                             model, clip, vae, steps, cfg,
                             controlnet, controlnet_strength, sampler_name, scheduler,
                             pbar=None):
        """Generate a tiled image composition with proper overlapping and seeding using ControlNet for outpainting."""

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Store a reference to the original controlnet - don't make copies
        original_controlnet = controlnet

        # Calculate total number of tiles
        total_tiles = grid_width * grid_height
        pbar = ProgressBar(total_tiles)

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

                # Calculate position in the final image
                pos_x = x * (tile_width - overlap_x)
                pos_y = y * (tile_height - overlap_y)

                # For first tile, simply generate
                if x == 0 and y == 0:
                    # Standard sampling without mask or controlnet
                    latent_height = tile_height // 8
                    latent_width = tile_width // 8

                    # Create an empty latent image as required by sample()
                    latent_image = torch.zeros([1, 4, latent_height, latent_width], device=device)

                    # Generate noise
                    noise = comfy.sample.prepare_noise(latent_image, seed, None)

                    # Sample using standard method
                    samples = comfy.sample.sample(
                        model,
                        noise,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        pos_cond,
                        neg_cond,
                        latent_image
                    )

                    # Decode
                    tile_tensor = vae.decode(samples)

                else:
                    # For subsequent tiles, use outpainting with ControlNet Union

                    # Create a completely black canvas for this tile
                    working_tensor = torch.zeros((1, tile_height, tile_width, 3), dtype=torch.float32)

                    # Create a mask where 1 = areas to generate, 0 = keep black areas
                    outpaint_mask = torch.ones((1, tile_height, tile_width, 1), dtype=torch.float32)

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

                    # 1. Get a properly sized empty latent shape
                    with torch.no_grad():
                        latent_shape = vae.encode(working_tensor).shape

                    # 2. Create empty latent tensor with correct shape
                    latent_image = torch.zeros(latent_shape, device=device)

                    # 3. Generate noise
                    noise = comfy.sample.prepare_noise(latent_image, current_seed, None)

                    # 4. Create the latent mask (1 = generate, 0 = keep)
                    mask_vae = outpaint_mask.permute(0, 3, 1, 2)
                    latent_mask = torch.nn.functional.interpolate(
                        mask_vae, size=(latent_shape[2], latent_shape[3]), mode="bilinear"
                    )

                    # 5. Apply ControlNet to conditioning
                    conditioning = self.apply_controlnet(
                        positive=pos_cond,
                        negative=neg_cond,
                        control_net=original_controlnet,
                        image=working_tensor,  # The image with content + black regions
                        strength=controlnet_strength,
                        start_percent=0.0,
                        end_percent=1.0,
                        vae=vae
                    )

                    # 6. Sample with the noise, empty latent, and mask
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

                    # Decode
                    tile_tensor = vae.decode(samples)

                # Store this tile
                individual_tensors.append(tile_tensor.clone())

                # Place the tile in the final composite
                h = min(tile_height, final_height - pos_y)
                w = min(tile_width, final_width - pos_x)

                final_tensor[0, pos_y:pos_y + h, pos_x:pos_x + w, :] = tile_tensor[0, :h, :w, :]

                # Update progress
                pbar.update(1)

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

            # Extract the prompts in the correct order, with more flexible position handling
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

# This part is needed for ComfyUI to recognize the nodes
NODE_CLASS_MAPPINGS = {
    "TiledImageGenerator": TiledImageGenerator,
}

# Add descriptions for the web UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageGenerator": "Tiled Image Generator (ControlNet Outpainting)",
}
