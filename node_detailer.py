import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

from .utils import feather_blend_latent, _compute_center_grid


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

        cols, rows, start_x, start_y = _compute_center_grid(W, H, tile_l, overlap_l)
        stride = tile_l - overlap_l
        total_tiles = (rows + 1) * (cols + 1)

        print(f"[LLMTileSequentialDetailer] Latent {W}x{H} | "
              f"tile_l={tile_l} overlap_l={overlap_l} stride={stride} | "
              f"grid cols={cols} rows={rows} ({total_tiles} tiles)")

        pbar = ProgressBar(total_tiles)

        for r in range(rows + 1):
            for c in range(cols + 1):
                y1 = max(0, start_y + r * stride)
                x1 = max(0, start_x + c * stride)
                y2 = min(H, y1 + tile_l)
                x2 = min(W, x1 + tile_l)

                if y2 <= y1 or x2 <= x1:
                    pbar.update(1)
                    continue

                tile_seed = seed + r * (cols + 1) + c
                tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

                noise = comfy.sample.prepare_noise(tile_latent, tile_seed, None)
                refined = comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler,
                    positive, negative, tile_latent,
                    denoise=denoise,
                )

                feather_blend_latent(
                    canvas, refined, y1, x1, overlap_l,
                    has_left=(x1 > 0),
                    has_top=(y1 > 0),
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
