# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import pytest

from encoders.models import ALL_MODELS, ALL_LAYERS


@pytest.fixture
def images(device):
    result = torch.empty((4, 3, 32, 32), dtype=torch.float32, device=device).normal_()
    return result.requires_grad_(True)


@pytest.mark.parametrize("model_class", ALL_MODELS)
class TestBackwardsThroughCheckpoints:
    @pytest.mark.parametrize("layer", ALL_LAYERS)
    def test_automatic(self, model, images, layer):
        results = dict(model.extract(images, layers=[layer], as_checkpoints=True))
        result = results[layer]
        loss = torch.nn.functional.mse_loss(result.mean(), result.new_zeros([]))

        assert images.grad is None
        loss.backward()
        assert images.grad is not None

    @pytest.mark.parametrize("layer", ALL_LAYERS)
    def test_detached(self, model, images, layer):
        results = dict(model.extract(images, layers=[layer], as_checkpoints=True))
        result = results[layer].detach().requires_grad_(True)

        loss = torch.nn.functional.mse_loss(result.mean(), result.new_zeros([]))

        assert result.grad is None
        loss.backward()
        assert result.grad is not None

        assert images.grad is None
        torch.autograd.backward(results[layer], [result.grad])
        assert images.grad is not None
