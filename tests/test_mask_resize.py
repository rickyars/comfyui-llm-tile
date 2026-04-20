import torch
import torch.nn.functional as F
import pytest


def resize_mask_to_latent(outpaint_mask, latent_h, latent_w):
    mask = outpaint_mask[:, :, :, 0]
    return F.interpolate(
        mask.unsqueeze(1),
        size=(latent_h, latent_w),
        mode='nearest'
    ).squeeze(1)


def test_output_shape():
    mask = torch.ones(1, 1024, 1024, 1)
    result = resize_mask_to_latent(mask, 128, 128)
    assert result.shape == (1, 128, 128)


def test_all_ones_preserved():
    mask = torch.ones(1, 1024, 1024, 1)
    result = resize_mask_to_latent(mask, 128, 128)
    assert result.all()


def test_all_zeros_preserved():
    mask = torch.zeros(1, 1024, 1024, 1)
    result = resize_mask_to_latent(mask, 128, 128)
    assert not result.any()


def test_left_overlap_region():
    overlap_x = 154
    mask = torch.ones(1, 1024, 1024, 1)
    mask[:, :, :overlap_x, :] = 0.0
    result = resize_mask_to_latent(mask, 128, 128)
    latent_overlap = round(128 * (overlap_x / 1024))
    assert result[:, :, :latent_overlap].sum() == 0
    assert result[:, :, latent_overlap:].sum() > 0


def test_asymmetric_canvas():
    mask = torch.ones(1, 1024, 1178, 1)
    mask[:, :, :154, :] = 0.0
    result = resize_mask_to_latent(mask, 128, 148)
    assert result.shape == (1, 128, 148)
