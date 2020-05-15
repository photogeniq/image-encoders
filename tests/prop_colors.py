# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

from hypothesis import given, strategies as H

import torch
from encoders import colors


def make_square_images(batch, height, width):
    return torch.empty((batch, 3, height, width), dtype=torch.float32).uniform_()


def Images() -> H.SearchStrategy[torch.Tensor]:
    return H.builds(
        make_square_images,
        batch=H.integers(min_value=1, max_value=8),
        height=H.integers(min_value=1, max_value=64),
        width=H.integers(min_value=1, max_value=64),
    )


@given(images=Images())
def test_xyz_lab_cycle_consistency(images):
    lab = colors.xyz2lab(images)
    xyz = colors.lab2xyz(lab)
    assert ((images - xyz).abs() < 1e-6).all()


@given(images=Images())
def test_rgb_xyz_cycle_consistency(images):
    xyz = colors.rgb2xyz(images)
    rgb = colors.xyz2rgb(xyz)
    assert ((images - rgb).abs() < 1e-5).all()


@given(images=Images())
def test_rgb_lab_cycle_consistency(images):
    lab = colors.rgb2lab(images)
    rgb = colors.lab2rgb(lab)
    assert ((images - rgb).abs() < 1e-5).all()
