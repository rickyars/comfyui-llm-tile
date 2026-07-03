# Prevents pytest from walking up to the parent ComfyUI pytest.ini,
# which would cause relative import errors in __init__.py
import sys
import os
import torch

# Stub out comfy.* so pure-function tests can import node_detailer_adaptive
# without needing a live ComfyUI installation
from types import ModuleType
import unittest.mock

def _make_comfy_stubs():
    comfy = ModuleType('comfy')
    comfy.sample = ModuleType('comfy.sample')
    comfy.model_management = ModuleType('comfy.model_management')
    comfy.samplers = ModuleType('comfy.samplers')
    comfy.utils = ModuleType('comfy.utils')

    # KSampler stub
    ksample = ModuleType('comfy.samplers.KSampler')
    ksample.SAMPLERS = []
    ksample.SCHEDULERS = []
    comfy.samplers.KSampler = ksample

    # ProgressBar stub: a no-op class (using MagicMock directly fails because
    # MagicMock(total) interprets the int arg as spec=int, leaving no .update())
    class _ProgressBar:
        def __init__(self, total): pass
        def update(self, value=1): pass
    comfy.utils.ProgressBar = _ProgressBar

    # prepare_noise — returns zeros matching the input shape
    comfy.sample.prepare_noise = unittest.mock.MagicMock(
        side_effect=lambda latent, seed, inds: torch.zeros_like(latent)
    )

    # fix_empty_latent_channels — real ComfyUI reshapes empty latents to the
    # model's channel count / rank; for tests a pass-through is enough
    comfy.sample.fix_empty_latent_channels = unittest.mock.MagicMock(
        side_effect=lambda model, latent: latent
    )

    # sample_custom — returns a clone of latent_image (arg index 7)
    comfy.sample.sample_custom = unittest.mock.MagicMock(
        side_effect=lambda model, noise, cfg, sampler, sigmas, pos, neg, latent, **kw: latent.clone()
    )

    # k_diffusion_sampling stub — two functions: one with eta, one without
    k_diff = ModuleType('comfy.samplers.k_diffusion_sampling')
    def _sample_with_eta(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0):
        pass
    def _sample_without_eta(model, x, sigmas, extra_args=None, callback=None, disable=None):
        pass
    k_diff.sample_euler_ancestral = _sample_with_eta
    k_diff.sample_euler = _sample_without_eta
    comfy.samplers.k_diffusion_sampling = k_diff
    sys.modules['comfy.samplers.k_diffusion_sampling'] = k_diff

    # ksampler — returns a mock whose extra_options matches what was passed
    def _mock_ksampler(name, extra_options=None):
        m = unittest.mock.MagicMock()
        m.extra_options = dict(extra_options) if extra_options else {}
        return m
    comfy.samplers.ksampler = _mock_ksampler

    # calculate_sigmas — returns a 21-element descending tensor (20 steps + endpoint)
    comfy.samplers.calculate_sigmas = unittest.mock.MagicMock(
        return_value=torch.linspace(14.6, 0.0, 21)
    )

    # model_management utilities
    comfy.model_management.soft_empty_cache = unittest.mock.MagicMock()
    comfy.model_management.intermediate_device = unittest.mock.MagicMock(
        return_value=torch.device('cpu')
    )
    comfy.model_management.intermediate_dtype = unittest.mock.MagicMock(
        return_value=torch.float32
    )

    sys.modules['comfy'] = comfy
    sys.modules['comfy.sample'] = comfy.sample
    sys.modules['comfy.model_management'] = comfy.model_management
    sys.modules['comfy.samplers'] = comfy.samplers
    sys.modules['comfy.utils'] = comfy.utils

_make_comfy_stubs()

# Add the project root to sys.path so tests can import from utils/ etc.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def pytest_collection(session):
    """Remove root __init__.py from collection."""
    # Prevents the root __init__.py from being imported as a test module
    pass
