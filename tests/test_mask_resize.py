import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils.image_utils import resize_mask_to_latent


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
