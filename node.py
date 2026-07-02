import torch
import comfy.sample
import comfy.controlnet
import comfy.model_management
from comfy.utils import ProgressBar

from .utils import parse_tile_prompts
from .utils import apply_controlnet_to_conditioning, blend_and_place_tile


def _apply_zimage_patch(model, model_patch, vae, working_tensor, keep_mask, strength):
    """Patch the model with the DiffSynth/Fun inpaint control (z-image, Qwen) for one tile.

    keep_mask: [H, W] with 1 where neighbour pixels are kept, 0 where new content is
    generated. The node's convention is white(1) = regenerate, so we pass its inverse.
    """
    from comfy_extras.nodes_model_patch import ZImageFunControlnet
    gen_mask = (1.0 - keep_mask).unsqueeze(0)  # (1, H, W); white = generate
    return ZImageFunControlnet().diffsynth_controlnet(
        model=model, model_patch=model_patch, vae=vae,
        inpaint_image=working_tensor, strength=strength, mask=gen_mask,
    )[0]


def _decode_tile(vae, samples):
    img = vae.decode(samples)
    if img.ndim == 5:  # video VAE (Krea2/Wan) -> single frame
        img = img[:, 0]
    return img[0].cpu()  # [gen_h8, gen_w8, 3]


class TiledImageGenerator:
    """
    LLM-directed tiled generation. Each tile generates on an expanded canvas that
    reaches into its already-generated neighbours' overlap; those neighbour pixels
    drive coherence via whichever mechanism is wired:

      * controlnet   -> ControlNet on the conditioning (SDXL, Flux/Krea2)
      * model_patch  -> DiffSynth/Fun inpaint patch on the model (z-image, Qwen)

    With neither wired, tiles generate independently (no coherence). Results are
    feather-placed into one canvas.
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
                "control_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01,
                                               "tooltip": "Strength of the coherence control (ControlNet or model patch)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "seamlessX": ("BOOLEAN", {"default": True, "tooltip": "If true, side of image will be seamless. (2+ tiles)"}),
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

    def generate_tiled_image(self, json_tile_prompts, global_positive, global_negative,
                             grid_width, grid_height, tile_width, tile_height,
                             overlap_percent, seed, model, clip, vae,
                             sampler_name, scheduler, steps, cfg, control_strength,
                             seamlessX, seamlessY,
                             controlnet=None, model_patch=None, control_net=None):

        # Accept the older `control_net` input name from previously-wired graphs.
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

        device = model.load_device

        pbar = ProgressBar(grid_width * grid_height)

        for y in range(grid_height):
            for x in range(grid_width):
                idx = y * grid_width + x
                current_prompt = tile_prompts[idx]
                current_seed = seed + idx
                final_pos_x = x * tile_width
                final_pos_y = y * tile_height

                has_left_neighbor = x > 0
                has_top_neighbor = y > 0
                # Both coherence mechanisms need the neighbour pixels, so both use the
                # expanded canvas + working tensor.
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
                combined_prompt = f"{current_prompt} {global_positive}" if global_positive else current_prompt
                pos_cond = clip.encode_from_tokens_scheduled(clip.tokenize(combined_prompt))
                neg_cond = clip.encode_from_tokens_scheduled(clip.tokenize(global_negative))

                positive, negative = pos_cond, neg_cond
                tile_model = model
                working_tensor = None
                keep_mask = None

                if coherence_active:
                    working_tensor = torch.zeros((1, gen_h8, gen_w8, 3), dtype=torch.float32)
                    keep_mask = torch.zeros((gen_h8, gen_w8), dtype=torch.float32)

                    if has_left_neighbor:
                        source_x = final_pos_x - overlap_x
                        source_start_y = final_pos_y
                        source_end_y = min(final_pos_y + tile_height, final_height)
                        source_height = source_end_y - source_start_y
                        target_start_y = overlap_y if has_top_neighbor else 0
                        copy_height = min(source_height, gen_h8 - target_start_y)
                        if copy_height > 0:
                            working_tensor[0,
                                target_start_y:target_start_y + copy_height, :overlap_x, :
                            ] = final_tensor[0,
                                source_start_y:source_start_y + copy_height,
                                source_x:source_x + overlap_x, :]
                            keep_mask[target_start_y:target_start_y + copy_height, :overlap_x] = 1.0

                    if seamlessX and x == grid_width - 1 and overlap_x > 0:
                        wrap_target_x = overlap_x + tile_width
                        source_start_y = final_pos_y
                        source_end_y = min(final_pos_y + tile_height, final_height)
                        source_height = source_end_y - source_start_y
                        target_start_y = overlap_y if has_top_neighbor else 0
                        copy_height = min(source_height, gen_h8 - target_start_y)
                        if copy_height > 0 and wrap_target_x + overlap_x <= gen_w8:
                            working_tensor[0,
                                target_start_y:target_start_y + copy_height,
                                wrap_target_x:wrap_target_x + overlap_x, :
                            ] = final_tensor[0,
                                source_start_y:source_start_y + copy_height, 0:overlap_x, :]
                            keep_mask[target_start_y:target_start_y + copy_height,
                                      wrap_target_x:wrap_target_x + overlap_x] = 1.0

                    if seamlessY and y == grid_height - 1 and overlap_y > 0:
                        wrap_target_y = overlap_y + tile_height
                        source_start_x = final_pos_x
                        source_end_x = min(final_pos_x + tile_width, final_width)
                        source_width = source_end_x - source_start_x
                        target_start_x = overlap_x if has_left_neighbor else 0
                        copy_width = min(source_width, gen_w8 - target_start_x)
                        if copy_width > 0 and wrap_target_y + overlap_y <= gen_h8:
                            working_tensor[0,
                                wrap_target_y:wrap_target_y + overlap_y,
                                target_start_x:target_start_x + copy_width, :
                            ] = final_tensor[0,
                                0:overlap_y, source_start_x:source_start_x + copy_width, :]
                            keep_mask[wrap_target_y:wrap_target_y + overlap_y,
                                      target_start_x:target_start_x + copy_width] = 1.0

                    if has_top_neighbor:
                        source_y = final_pos_y - overlap_y
                        source_start_x = final_pos_x
                        source_end_x = min(final_pos_x + tile_width, final_width)
                        source_width = source_end_x - source_start_x
                        target_start_x = overlap_x if has_left_neighbor else 0
                        copy_width = min(source_width, gen_w8 - target_start_x)
                        if copy_width > 0:
                            working_tensor[0,
                                :overlap_y, target_start_x:target_start_x + copy_width, :
                            ] = final_tensor[0,
                                source_y:source_y + overlap_y,
                                source_start_x:source_start_x + copy_width, :]
                            keep_mask[:overlap_y, target_start_x:target_start_x + copy_width] = 1.0

                    if has_left_neighbor and has_top_neighbor:
                        corner_source_x = final_pos_x - overlap_x
                        corner_source_y = final_pos_y - overlap_y
                        working_tensor[0, :overlap_y, :overlap_x, :] = final_tensor[0,
                            corner_source_y:corner_source_y + overlap_y,
                            corner_source_x:corner_source_x + overlap_x, :]
                        keep_mask[:overlap_y, :overlap_x] = 1.0

                    if model_patch is not None:
                        tile_model = _apply_zimage_patch(
                            model, model_patch, vae, working_tensor, keep_mask, control_strength)
                    elif controlnet is not None:
                        positive, negative = apply_controlnet_to_conditioning(
                            positive=pos_cond, negative=neg_cond, control_net=controlnet,
                            image=working_tensor, strength=control_strength,
                            start_percent=0.0, end_percent=1.0, vae=vae)

                # 16-channel (z-image/Krea2) and video-latent formats need the right shape.
                latent_image = comfy.sample.fix_empty_latent_channels(
                    model, torch.zeros((1, 4, gen_h8 // 8, gen_w8 // 8), device=device))
                noise = comfy.sample.prepare_noise(latent_image, current_seed, None)
                samples = comfy.sample.sample(
                    tile_model, noise, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent_image)

                decoded = _decode_tile(vae, samples)  # [gen_h8, gen_w8, 3]

                # For the model_patch path the control's reconstruction of the kept
                # overlap can return flat/grey; force those pixels back to the real
                # neighbour so the feather blends real content, not grey.
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

                comfy.model_management.soft_empty_cache()
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
