import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from utils.sampling_utils import check_eta_support, prepare_noise_typed, build_tile_sampler


# --- check_eta_support ---

def test_check_eta_support_true_for_sampler_with_eta():
    # conftest stubs k_diffusion_sampling.sample_euler_ancestral with eta param
    assert check_eta_support("euler_ancestral") is True


def test_check_eta_support_false_for_sampler_without_eta():
    # conftest stubs k_diffusion_sampling.sample_euler without eta param
    assert check_eta_support("euler") is False


def test_check_eta_support_false_for_unknown_sampler():
    assert check_eta_support("nonexistent_sampler_xyz") is False


# --- prepare_noise_typed ---

def test_prepare_noise_typed_gaussian_delegates_to_comfy():
    import comfy.sample
    tile = torch.zeros(1, 4, 8, 8)
    comfy.sample.prepare_noise.reset_mock()
    prepare_noise_typed(tile, seed=42, noise_type="gaussian", sigma_min=0.03, sigma_max=14.6)
    comfy.sample.prepare_noise.assert_called_once_with(tile, 42, None)


def test_prepare_noise_typed_returns_tensor_matching_input_shape():
    tile = torch.zeros(1, 4, 8, 8)
    result = prepare_noise_typed(tile, seed=42, noise_type="gaussian", sigma_min=0.03, sigma_max=14.6)
    assert isinstance(result, torch.Tensor)
    assert result.shape == tile.shape


def test_prepare_noise_typed_unknown_type_falls_back_to_gaussian():
    # With no RES4LYF present in test env, any non-gaussian type falls back
    import comfy.sample
    tile = torch.zeros(1, 4, 8, 8)
    comfy.sample.prepare_noise.reset_mock()
    prepare_noise_typed(tile, seed=0, noise_type="brownian", sigma_min=0.03, sigma_max=14.6)
    # Should call comfy fallback (RES4LYF not available in test env)
    comfy.sample.prepare_noise.assert_called_once()


# --- build_tile_sampler ---

def test_build_tile_sampler_includes_eta_when_supported():
    sampler = build_tile_sampler("euler_ancestral", tile_eta=0.7, eta_supported=True)
    assert sampler.extra_options == {"eta": 0.7}


def test_build_tile_sampler_omits_eta_when_unsupported():
    sampler = build_tile_sampler("euler", tile_eta=0.7, eta_supported=False)
    assert sampler.extra_options == {}


def test_build_tile_sampler_eta_zero_still_passes_when_supported():
    sampler = build_tile_sampler("euler_ancestral", tile_eta=0.0, eta_supported=True)
    assert sampler.extra_options == {"eta": 0.0}
