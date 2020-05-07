# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import pytest

from encoders.models import ALL_MODELS, ALL_LAYERS


@pytest.fixture
def images(device):
    return torch.empty((2, 3, 32, 32), dtype=torch.float32, device=device).normal_()


@pytest.mark.parametrize("model_class", ALL_MODELS)
class TestForward:
    @pytest.mark.parametrize("layer", ALL_LAYERS)
    def test_extract_one_layer(self, model, images, layer):
        lid = int(layer[0])
        r = dict(model.extract(images, layers=[layer]))
        assert r[layer].shape[0] == images.shape[0]
        assert r[layer].shape[1] == model.CONFIG[lid - 1].filters
        for i in (2, 3):
            assert r[layer].dtype == torch.float32
            assert r[layer].shape[i] == images.shape[i] // 2 ** (lid - 1)

    def test_extract_all_layers(self, model, images):
        r = dict(model.extract(images, layers=ALL_LAYERS))
        assert set(r.keys()) == set(ALL_LAYERS)

    @pytest.mark.cuda
    @pytest.mark.parametrize("layer", ALL_LAYERS)
    def test_extract_half_precision(self, model, images, layer):
        lid = int(layer[0])
        r = dict(model.half().extract(images.half(), layers=[layer]))
        assert r[layer].shape[0] == images.shape[0]
        assert r[layer].shape[1] == model.CONFIG[lid - 1].filters
        for i in (2, 3):
            assert r[layer].dtype == torch.float16
            assert r[layer].shape[i] == images.shape[i] // 2 ** (lid - 1)
