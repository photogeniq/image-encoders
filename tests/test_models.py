# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch
import pytest

from encoders.models import ALL_MODELS


@pytest.mark.parametrize("model_class", ALL_MODELS)
class TestLifeCycle:
    def test_construct_default(self, model_class):
        model = model_class(pretrained=False)
        del model

    def test_train_eval(self, model_class):
        model = model_class(pretrained=False)
        model.train()
        model.eval()

    def test_to_device(self, device, model_class):
        model = model_class(pretrained=False).to(device)
        del model

    @pytest.mark.cuda
    def test_half_precision(self, device, model_class):
        model = model_class(pretrained=False).to(device).half()
        del model
