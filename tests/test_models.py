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


@pytest.mark.parametrize("model_class", ALL_MODELS)
class TestPretrained:
    def test_construct_pretrained(self, model_class):
        model = model_class(pretrained=True)
        del model

    def test_preload_float32(self, model_class):
        m1 = model_class(pretrained=True)
        m2 = model_class(pretrained=True)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert p1.dtype == torch.float32
            assert (p1 == p2).all()

    @pytest.mark.cuda
    def test_preload_float16(self, model_class, device):
        m1 = model_class(pretrained=True).to(device=device, dtype=torch.float16)
        m2 = model_class(pretrained=True).to(device=device, dtype=torch.float16)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert p1.dtype == torch.float16
            assert (p1 == p2).all()
