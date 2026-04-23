import torch
import comfy.sample
import comfy.controlnet
from comfy.utils import ProgressBar

from .utils import parse_tile_prompts
from .utils import apply_controlnet_to_conditioning, blend_and_place_tile


class TiledImageGenerator:
    """
    ComfyUI node that generates a tiled image composition. Each tile generates at
    exactly tile_width x tile_height. Neighbor overlap pixels are passed to ControlNet
    as a reference image. Tiles are blended into the canvas with linear feathering.
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
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "seamlessX": ("BOOLEAN", {"default": True, "tooltip": "If true, side of image will be seamless. (2+ tiles)"}),
                "seamlessY": ("BOOLEAN", {"default": False, "tooltip": "If true, top/bottom of image will be seamless. (2+ tiles)"}),
            },
            "optional": {
                "controlnet": ("CONTROL_NET",),
                "controlnet_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("composite_image", "individual_tiles")
    FUNCTION = "generate_tiled_image"
    CATEGORY = "image/generation"

    def generate_tiled_image(self, json_tile_prompts, global_positive, global_negative,
                             grid_width, grid_height, tile_width, tile_height,
                             overlap_percent, seed, model, clip, vae,
                             sampler_name, scheduler, steps, cfg, seamlessX, seamlessY,
                             controlnet=None, controlnet_strength=0.7):

        tile_prompts = parse_tile_prompts(json_tile_prompts, grid_width, grid_height)

        overlap_x = int(tile_width * overlap_percent)
        overlap_y = int(tile_height * overlap_percent)
        final_width = tile_width * grid_width
        final_height = tile_height * grid_height

        print(f"Final output size: {final_width} x {final_height}")
        print(f"Tile size: {tile_width} x {tile_height}")
        print(f"Grid: {grid_width} x {grid_height}")
        print(f"Overlap: {overlap_x} x {overlap_y} pixels ({overlap_percent * 100}%)")

        # Fixed generation canvas — always exactly tile dimensions
        gen_width = ((tile_width + 7) // 8) * 8
        gen_height = ((tile_height + 7) // 8) * 8

        final_tensor = torch.zeros((1, final_height, final_width, 3), dtype=torch.float32)
        individual_tiles = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        pbar = ProgressBar(grid_width * grid_height)

        for y in range(grid_height):
            for x in range(grid_width):
                idx = y * grid_width + x
                current_prompt = tile_prompts[idx]
                current_seed = seed + idx
                final_pos_x = x * tile_width
                final_pos_y = y * tile_height

                print(f"Tile ({x + 1},{y + 1}) generation canvas: {gen_width} x {gen_height}")
                print(f"Generating tile ({x + 1},{y + 1}) with prompt: {current_prompt}")

                combined_prompt = f"{current_prompt} {global_positive}" if global_positive else current_prompt
                pos_cond = clip.encode_from_tokens_scheduled(clip.tokenize(combined_prompt))
                neg_cond = clip.encode_from_tokens_scheduled(clip.tokenize(global_negative))

                # Build working tensor: overlap pixels from neighbors for ControlNet reference
                working_tensor = torch.zeros((1, gen_height, gen_width, 3), dtype=torch.float32)
                has_left_neighbor = x > 0
                has_top_neighbor = y > 0

                if has_left_neighbor:
                    source_x = final_pos_x - overlap_x
                    if source_x >= 0 and source_x + overlap_x <= final_width:
                        source_start_y = final_pos_y
                        source_end_y = min(final_pos_y + tile_height, final_height)
                        source_height = source_end_y - source_start_y
                        target_start_y = overlap_y if has_top_neighbor else 0
                        copy_height = min(source_height, gen_height - target_start_y)
                        if copy_height > 0:
                            working_tensor[0,
                                target_start_y:target_start_y + copy_height, :overlap_x, :
                            ] = final_tensor[0,
                                source_start_y:source_start_y + copy_height,
                                source_x:source_x + overlap_x, :]

                if seamlessX and x == grid_width - 1 and overlap_x > 0:
                    wrap_target_x = gen_width - overlap_x
                    source_start_y = final_pos_y
                    source_end_y = min(final_pos_y + tile_height, final_height)
                    source_height = source_end_y - source_start_y
                    target_start_y = overlap_y if has_top_neighbor else 0
                    copy_height = min(source_height, gen_height - target_start_y)
                    if copy_height > 0 and wrap_target_x >= 0:
                        working_tensor[0,
                            target_start_y:target_start_y + copy_height,
                            wrap_target_x:wrap_target_x + overlap_x, :
                        ] = final_tensor[0,
                            source_start_y:source_start_y + copy_height, 0:overlap_x, :]

                if seamlessY and y == grid_height - 1 and overlap_y > 0:
                    wrap_target_y = gen_height - overlap_y
                    source_start_x = final_pos_x
                    source_end_x = min(final_pos_x + tile_width, final_width)
                    source_width = source_end_x - source_start_x
                    target_start_x = overlap_x if has_left_neighbor else 0
                    copy_width = min(source_width, gen_width - target_start_x)
                    if copy_width > 0 and wrap_target_y >= 0:
                        working_tensor[0,
                            wrap_target_y:wrap_target_y + overlap_y,
                            target_start_x:target_start_x + copy_width, :
                        ] = final_tensor[0,
                            0:overlap_y, source_start_x:source_start_x + copy_width, :]

                if has_top_neighbor:
                    source_y = final_pos_y - overlap_y
                    if source_y >= 0 and source_y + overlap_y <= final_height:
                        source_start_x = final_pos_x
                        source_end_x = min(final_pos_x + tile_width, final_width)
                        source_width = source_end_x - source_start_x
                        target_start_x = overlap_x if has_left_neighbor else 0
                        copy_width = min(source_width, gen_width - target_start_x)
                        if copy_width > 0:
                            working_tensor[0,
                                :overlap_y, target_start_x:target_start_x + copy_width, :
                            ] = final_tensor[0,
                                source_y:source_y + overlap_y,
                                source_start_x:source_start_x + copy_width, :]

                if has_left_neighbor and has_top_neighbor:
                    corner_source_x = final_pos_x - overlap_x
                    corner_source_y = final_pos_y - overlap_y
                    if (corner_source_x >= 0 and corner_source_y >= 0 and
                            corner_source_x + overlap_x <= final_width and
                            corner_source_y + overlap_y <= final_height):
                        working_tensor[0, :overlap_y, :overlap_x, :] = final_tensor[0,
                            corner_source_y:corner_source_y + overlap_y,
                            corner_source_x:corner_source_x + overlap_x, :]

                with torch.no_grad():
                    latent_shape = vae.encode(working_tensor).shape
                latent_image = torch.zeros(latent_shape, device=device)

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

                noise = comfy.sample.prepare_noise(latent_image, current_seed, None)
                samples = comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler,
                    conditioning[0], conditioning[1], latent_image
                )

                extracted_tile = vae.decode(samples)[0][:tile_height, :tile_width, :]
                individual_tiles.append(extracted_tile.unsqueeze(0))

                blend_and_place_tile(
                    final_tensor, extracted_tile,
                    pos_x=final_pos_x, pos_y=final_pos_y,
                    tile_width=tile_width, tile_height=tile_height,
                    overlap_x=overlap_x, overlap_y=overlap_y,
                    has_left=has_left_neighbor, has_top=has_top_neighbor,
                    controlnet_active=False
                )
                pbar.update(1)

        tile_batch = torch.cat(individual_tiles, dim=0) if individual_tiles else \
            torch.zeros((1, tile_height, tile_width, 3), dtype=torch.float32)

        if seamlessX and overlap_x > 0:
            final_tensor = final_tensor[:, :, :final_width - overlap_x, :]
        if seamlessY and overlap_y > 0:
            final_tensor = final_tensor[:, :final_height - overlap_y, :, :]

        return final_tensor, tile_batch


NODE_CLASS_MAPPINGS = {
    "TiledImageGenerator": TiledImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledImageGenerator": "Tiled Image Generator",
}
