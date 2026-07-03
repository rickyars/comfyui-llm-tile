import torch
import comfy.sample
import comfy.model_management
from comfy.utils import ProgressBar

from .utils import parse_tile_prompts
from .utils import apply_controlnet_to_conditioning, blend_and_place_tile, build_working_tensor
from .node import _apply_zimage_patch, _decode_tile


class TiledImageGeneratorAdvanced:
    """
    Advanced tiled generator (custom sampler/guider/sigmas). Same coherence model as
    TiledImageGenerator: neighbour pixels drive a ControlNet (conditioning) or a
    DiffSynth/Fun inpaint model patch (z-image/Qwen), feather-placed into one canvas.
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
                "control_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01}),
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "seamlessX": ("BOOLEAN", {"default": True, "tooltip": "If true, left/right of image will be seamless. (2+ tiles)"}),
                "seamlessY": ("BOOLEAN", {"default": False, "tooltip": "If true, top/bottom of image will be seamless. (2+ tiles)"}),
            },
            "optional": {
                "controlnet": ("CONTROL_NET",),
                "model_patch": ("MODEL_PATCH",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("composite_image", "individual_tiles")
    FUNCTION = "generate_tiled_image"
    CATEGORY = "image/generation"

    def generate_tiled_image(self, json_tile_prompts, grid_width, grid_height,
                             tile_width, tile_height, overlap_percent, control_strength,
                             noise, guider, sampler, sigmas, clip, vae,
                             seed, seamlessX, seamlessY,
                             controlnet=None, model_patch=None, control_net=None):

        if controlnet is None:
            controlnet = control_net

        tile_prompts = parse_tile_prompts(json_tile_prompts, grid_width, grid_height)

        overlap_x = int(tile_width * overlap_percent)
        overlap_y = int(tile_height * overlap_percent)
        final_width = tile_width * grid_width
        final_height = tile_height * grid_height

        mode = ("model_patch" if model_patch is not None
                else "controlnet" if controlnet is not None else "none")
        print(f"Final output size: {final_width} x {final_height}")
        print(f"Tile size: {tile_width} x {tile_height} | Grid: {grid_width} x {grid_height}")
        print(f"Overlap: {overlap_x} x {overlap_y} pixels ({overlap_percent * 100}%) | coherence: {mode}")

        final_tensor = torch.zeros((1, final_height, final_width, 3), dtype=torch.float32)
        individual_tiles = []

        orig_patcher = guider.model_patcher
        device = orig_patcher.load_device

        pbar = ProgressBar(grid_width * grid_height)

        try:
            for y in range(grid_height):
                for x in range(grid_width):
                    idx = y * grid_width + x
                    current_prompt = tile_prompts[idx]
                    current_seed = seed + idx
                    final_pos_x = x * tile_width
                    final_pos_y = y * tile_height

                    has_left_neighbor = x > 0
                    has_top_neighbor = y > 0
                    coherence_active = controlnet is not None or model_patch is not None

                    if coherence_active:
                        gen_w = (tile_width
                                 + (overlap_x if has_left_neighbor else 0)
                                 + (overlap_x if seamlessX and x == grid_width - 1 else 0))
                        gen_h = (tile_height
                                 + (overlap_y if has_top_neighbor else 0)
                                 + (overlap_y if seamlessY and y == grid_height - 1 else 0))
                    else:
                        gen_w, gen_h = tile_width, tile_height

                    gen_w8 = ((gen_w + 7) // 8) * 8
                    gen_h8 = ((gen_h + 7) // 8) * 8

                    print(f"Tile ({x+1},{y+1}) gen canvas: {gen_w8}x{gen_h8}, canvas pos: ({final_pos_x},{final_pos_y})")
                    print(f"Generating tile ({x+1},{y+1}) prompt: {current_prompt}")

                    # Per-tile encoding: a shared neg_cond corrupts SDXL pooled_output across tiles.
                    pos_cond = clip.encode_from_tokens_scheduled(clip.tokenize(current_prompt))
                    neg_cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))

                    positive, negative = pos_cond, neg_cond
                    working_tensor = None
                    keep_mask = None
                    guider.model_patcher = orig_patcher

                    if coherence_active:
                        working_tensor, keep_mask = build_working_tensor(
                            final_tensor, final_pos_x, final_pos_y,
                            tile_width, tile_height, overlap_x, overlap_y,
                            gen_w8, gen_h8, has_left_neighbor, has_top_neighbor,
                            wrap_x=(seamlessX and x == grid_width - 1),
                            wrap_y=(seamlessY and y == grid_height - 1))

                        if model_patch is not None:
                            guider.model_patcher = _apply_zimage_patch(
                                orig_patcher, model_patch, vae, working_tensor, keep_mask, control_strength)
                        elif controlnet is not None:
                            positive, negative = apply_controlnet_to_conditioning(
                                positive=pos_cond, negative=neg_cond, control_net=controlnet,
                                image=working_tensor, strength=control_strength,
                                start_percent=0.0, end_percent=1.0, vae=vae)

                    latent_image = comfy.sample.fix_empty_latent_channels(
                        orig_patcher, torch.zeros((1, 4, gen_h8 // 8, gen_w8 // 8), device=device))
                    tile_noise = noise.generate_noise({"samples": latent_image})

                    guider.set_conds(positive, negative)
                    samples = guider.sample(
                        tile_noise, latent_image, sampler, sigmas,
                        denoise_mask=None, disable_pbar=False, seed=current_seed)

                    decoded = _decode_tile(vae, samples)  # [gen_h8, gen_w8, 3]

                    if model_patch is not None and keep_mask is not None:
                        k = keep_mask.unsqueeze(-1)
                        decoded = (1.0 - k) * decoded + k * working_tensor[0]

                    start_x = (overlap_x if has_left_neighbor else 0) if coherence_active else 0
                    start_y = (overlap_y if has_top_neighbor else 0) if coherence_active else 0
                    individual_tiles.append(
                        decoded[start_y:start_y + tile_height, start_x:start_x + tile_width, :].unsqueeze(0))

                    blend_and_place_tile(
                        final_tensor, decoded, final_pos_x, final_pos_y,
                        tile_width, tile_height, overlap_x, overlap_y,
                        has_left_neighbor, has_top_neighbor, coherence_active)

                    # Flushing the CUDA cache every tile costs real time and mostly frees
                    # memory the next tile immediately re-allocates; throttle it.
                    if (idx + 1) % 4 == 0:
                        comfy.model_management.soft_empty_cache()
                    pbar.update(1)
        finally:
            guider.model_patcher = orig_patcher

        comfy.model_management.soft_empty_cache()

        tile_batch = torch.cat(individual_tiles, dim=0) if individual_tiles else \
            torch.zeros((1, tile_height, tile_width, 3), dtype=torch.float32)

        if seamlessX and overlap_x > 0:
            final_tensor = final_tensor[:, :, :final_width - overlap_x, :]
        if seamlessY and overlap_y > 0:
            final_tensor = final_tensor[:, :final_height - overlap_y, :, :]

        return final_tensor, tile_batch


NODE_CLASS_MAPPINGS = {
    "TiledImageGeneratorAdvanced": TiledImageGeneratorAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageGeneratorAdvanced": "Tiled Image Generator Advanced",
}
