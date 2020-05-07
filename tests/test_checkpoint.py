# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import pytest

from encoders.models import ALL_MODELS, ALL_LAYERS


@pytest.fixture
def images(device):
    result = torch.empty((1, 3, 32, 32), dtype=torch.float32, device=device).normal_()
    return result.requires_grad_(True)


@pytest.mark.cuda
@pytest.mark.parametrize("model_class", ALL_MODELS)
class TestMemoryUsage:
    @pytest.mark.parametrize("layer", ALL_LAYERS)
    def test_one_layer(self, model, images, layer):
        allocated = torch.cuda.memory_allocated()
        results = dict(model.extract(images, layers=[layer], as_checkpoints=True))
        delta = torch.cuda.memory_allocated() - allocated

        size = sum(r.storage().size() for r in results.values())
        assert (size * 4) / delta == 1.0

    def test_all_layers(self, model, images):
        allocated = torch.cuda.memory_allocated()
        results = dict(model.extract(images, layers=ALL_LAYERS, as_checkpoints=True))
        delta = torch.cuda.memory_allocated() - allocated

        size = sum(r.storage().size() for r in results.values())
        assert (size * 4) / delta == 1.0
