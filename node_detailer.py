import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

from .utils import feather_blend_latent, _compute_center_grid, _compute_tile_coords


class LLMTileSequentialDetailer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "upscaled_latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.01}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("refined_latent",)
    FUNCTION = "detail"
    CATEGORY = "image/generation"

    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler, denoise, tile_size, overlap):

        canvas = upscaled_latent["samples"].clone()
        _, _, H, W = canvas.shape

        tile_l = tile_size // 8
        overlap_l = overlap // 8
        if overlap_l >= tile_l:
            overlap_l = tile_l // 2
            print(f"[LLMTileSequentialDetailer] Warning: overlap clamped to "
                  f"{overlap_l * 8}px (overlap must be < tile_size)")

        cols, rows = _compute_center_grid(W, H, tile_l, overlap_l)
        stride = tile_l - overlap_l
        total_tiles = (rows + 1) * (cols + 1)

        print(f"[LLMTileSequentialDetailer] Latent {W}x{H} | "
              f"tile_l={tile_l} overlap_l={overlap_l} stride={stride} | "
              f"grid cols={cols} rows={rows} ({total_tiles} tiles)")

        pbar = ProgressBar(total_tiles)
        tile_coords = _compute_tile_coords(W, H, tile_l, cols, rows)
        n_cols = cols + 1

        for tile_idx, (y1, x1, y2, x2) in enumerate(tile_coords):
            r = tile_idx // n_cols
            c = tile_idx % n_cols
            tile_seed = seed + tile_idx
            tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

            noise = comfy.sample.prepare_noise(tile_latent, tile_seed, None)
            refined = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler,
                positive, negative, tile_latent,
                denoise=denoise,
            )

            feather_blend_latent(
                canvas, refined, y1, x1, overlap_l,
                has_left=(c > 0 and x1 < tile_coords[tile_idx - 1][3]),
                has_top=(r > 0 and y1 < tile_coords[tile_idx - n_cols][2]),
            )

            comfy.model_management.soft_empty_cache()
            pbar.update(1)

        return ({"samples": canvas},)


NODE_CLASS_MAPPINGS = {
    "LLMTileSequentialDetailer": LLMTileSequentialDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMTileSequentialDetailer": "Tiled Image Detailer",
}
