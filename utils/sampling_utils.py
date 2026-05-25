import inspect
import comfy.sample
import comfy.samplers

try:
    from RES4LYF.beta.noise_classes import (
        NOISE_GENERATOR_CLASSES_SIMPLE,
        NOISE_GENERATOR_NAMES_SIMPLE,
    )
    _RES4LYF_NOISE = True
except ImportError:
    NOISE_GENERATOR_NAMES_SIMPLE = ("gaussian",)
    _RES4LYF_NOISE = False


def check_eta_support(sampler_name):
    """Return True if the named sampler function accepts an eta keyword argument."""
    fn = getattr(comfy.samplers.k_diffusion_sampling, f"sample_{sampler_name}", None)
    if fn is None:
        return False
    return 'eta' in inspect.signature(fn).parameters


def prepare_noise_typed(tile_latent, seed, noise_type, sigma_min, sigma_max):
    """Generate initial tile noise using the specified distribution.

    Falls back to standard Gaussian if RES4LYF is not installed or noise_type is 'gaussian'.
    """
    if not _RES4LYF_NOISE or noise_type == "gaussian":
        return comfy.sample.prepare_noise(tile_latent, seed, None)
    cls = NOISE_GENERATOR_CLASSES_SIMPLE[noise_type]
    gen = cls(x=tile_latent, seed=seed, sigma_min=sigma_min, sigma_max=sigma_max)
    return gen(sigma=float(sigma_max), sigma_next=float(sigma_min))


def build_tile_sampler(sampler_name, tile_eta, eta_supported):
    """Build a KSAMPLER for one tile, injecting eta into extra_options if supported."""
    extra_opts = {'eta': tile_eta} if eta_supported else {}
    return comfy.samplers.ksampler(sampler_name, extra_options=extra_opts)
