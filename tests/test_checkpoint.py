# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import pytest

from encoders.models import ALL_MODELS, ALL_LAYERS


@pytest.fixture
def images(device):
    return torch.empty((1, 3, 256, 256), dtype=torch.float16, device=device).normal_()


@pytest.mark.cuda
@pytest.mark.parametrize("model_class", ALL_MODELS)
class TestMemoryUsage:
    @pytest.mark.parametrize("layer", ALL_LAYERS)
    def test_checkpoint(self, model_class, device, images, layer):
        model = model_class(pretrained=False).to(device=device, dtype=torch.float16)
        images.requires_grad_(True)

        allocated = torch.cuda.memory_allocated()
        results = dict(model.extract(images, layers=[layer], as_checkpoints=True))
        delta = torch.cuda.memory_allocated() - allocated

        size = sum(r.storage().size() for r in results.values())
        assert (size * 2) / delta == 1.0
