import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from comfy.utils import ProgressBar

# Support both relative imports (in package) and direct imports (in tests)
if __package__:
    from .utils import (
        feather_blend_latent, _compute_center_grid, _compute_tile_coords,
        check_eta_support, prepare_noise_typed, build_tile_sampler,
        NOISE_GENERATOR_NAMES_SIMPLE,
    )
else:
    from utils import (
        feather_blend_latent, _compute_center_grid, _compute_tile_coords,
        check_eta_support, prepare_noise_typed, build_tile_sampler,
        NOISE_GENERATOR_NAMES_SIMPLE,
    )


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
                "crop_to_tiles": ("BOOLEAN", {"default": False}),
                "noise_type": (NOISE_GENERATOR_NAMES_SIMPLE, {"default": "gaussian"}),
                "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                  "tooltip": "SDE noise injection per step. 0 = deterministic ODE. Only applies to samplers that support eta (e.g. euler_ancestral, rk_beta)."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("refined_latent",)
    FUNCTION = "detail"
    CATEGORY = "image/generation"

    def detail(self, model, upscaled_latent, positive, negative,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               tile_size, overlap, crop_to_tiles, noise_type, eta):

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

        model_sampling = model.get_model_object("model_sampling")
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)
        eta_supported = check_eta_support(sampler_name)
        if eta > 0.0 and not eta_supported:
            print(f"[LLMTileSequentialDetailer] Warning: '{sampler_name}' does not support "
                  f"eta; eta will be ignored. Use 'rk_beta' for full RES4LYF eta control.")

        if denoise >= 1.0:
            tile_sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        else:
            new_steps = int(steps / denoise)
            tile_sigmas = comfy.samplers.calculate_sigmas(
                model_sampling, scheduler, new_steps)[-(steps + 1):]
        tile_sigmas = tile_sigmas.to(model.load_device)
        sampler = build_tile_sampler(sampler_name, eta, eta_supported)

        pbar = ProgressBar(total_tiles)
        tile_coords = _compute_tile_coords(W, H, tile_l, cols, rows, overlap_l)
        n_cols = cols + 1

        for tile_idx, (y1, x1, y2, x2) in enumerate(tile_coords):
            r = tile_idx // n_cols
            c = tile_idx % n_cols
            tile_seed = seed + tile_idx
            tile_latent = canvas[:, :, y1:y2, x1:x2].clone()

            noise = prepare_noise_typed(tile_latent, tile_seed, noise_type, sigma_min, sigma_max)
            refined = comfy.sample.sample_custom(
                model, noise, cfg, sampler, tile_sigmas,
                positive, negative, tile_latent,
            )

            feather_blend_latent(
                canvas, refined, y1, x1, overlap_l,
                has_left=(c > 0 and x1 < tile_coords[tile_idx - 1][3]),
                has_top=(r > 0 and y1 < tile_coords[tile_idx - n_cols][2]),
            )

            comfy.model_management.soft_empty_cache()
            pbar.update(1)

        if crop_to_tiles:
            y1_c, x1_c = tile_coords[0][0], tile_coords[0][1]
            y2_c, x2_c = tile_coords[-1][2], tile_coords[cols][3]
            canvas = canvas[:, :, y1_c:y2_c, x1_c:x2_c]

        return ({"samples": canvas},)


NODE_CLASS_MAPPINGS = {
    "LLMTileSequentialDetailer": LLMTileSequentialDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMTileSequentialDetailer": "Tiled Image Detailer",
}
