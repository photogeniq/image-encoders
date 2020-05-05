# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import pytest

from encoders.models import *


ALL_MODELS = [VGG19, VGG16, VGG13, VGG11, ThinetSmall, ThinetTiny]
ALL_LAYERS = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("model_class", ALL_MODELS)
class TestLifeCycle:
    def test_construct_default(self, model_class):
        model = model_class()
        del model

    def test_train_eval(self, model_class):
        model = model_class()
        model.train()
        model.eval()

    def test_to_device(self, device, model_class):
        model = model_class().to(device)
        del model

    @pytest.mark.cuda
    def test_half_precision(self, device, model_class):
        model = model_class().to(device).half()
        del model


@pytest.fixture
def images(device):
    return torch.empty((2, 3, 32, 32), dtype=torch.float32, device=device).normal_()


@pytest.mark.parametrize("model_class", ALL_MODELS)
class TestForward:
    @pytest.mark.parametrize("layer", ALL_LAYERS)
    def test_extract_one_layer(self, model_class, device, images, layer):
        lid = f"{layer}_1"
        model = model_class().to(device)
        r = dict(model.extract(images, layers=[lid]))
        assert r[lid].shape[0] == images.shape[0]
        assert r[lid].shape[1] == model.CONFIG[layer - 1].filters
        for i in (2, 3):
            assert r[lid].shape[i] == images.shape[i] // 2 ** (layer - 1)

    def test_extract_all_layers(self, model_class, device, images):
        layers = [f"{l}_1" for l in ALL_LAYERS]
        model = model_class().to(device)
        r = dict(model.extract(images, layers=layers))
        assert set(r.keys()) == set(layers)

    @pytest.mark.cuda
    @pytest.mark.parametrize("layer", ALL_LAYERS)
    def test_extract_half_precision(self, model_class, device, images, layer):
        lid = f"{layer}_1"
        model = model_class().half().to(device)
        r = dict(model.extract(images.half(), layers=[lid]))
        assert r[lid].shape[0] == images.shape[0]
        assert r[lid].shape[1] == model.CONFIG[layer - 1].filters
        for i in (2, 3):
            assert r[lid].shape[i] == images.shape[i] // 2 ** (layer - 1)
