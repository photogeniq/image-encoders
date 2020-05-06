# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import functools
import collections

import torch
import torch.utils.checkpoint as tcp

from . import io
from . import convert


class ModuleConfig:
    def __init__(self, f, b):
        self.filters = f
        self.blocks = b


class Encoder(torch.nn.Module):
    def __init__(
        self, block_type, pool_type, input_type, in_channels=3, pretrained=True
    ):
        super(Encoder, self).__init__()

        blocks = self.make_blocks(self.CONFIG, block_type, pool_type, in_channels)
        if input_type == "RGB":
            blocks = [("0_1", convert.NormalizeRGB())] + list(blocks)

        self.features = torch.nn.Sequential(collections.OrderedDict(blocks))

        if pretrained is True:
            self.load_pretrained(self.FILENAME, self.HEXDIGEST)

    def load_pretrained(self, model: str, hexdigest: str):
        fullpath = io.download_to_file(model, hexdigest)
        self.load_state_dict(torch.load(fullpath), strict=False)

    def make_blocks(self, config, block_type, pool_type, in_channels):
        previous = in_channels
        for octave, module in enumerate(config):
            for block in range(module.blocks):
                yield f"{octave+1}_{block+1}", block_type(previous, module.filters)
                if block == module.blocks - 1:
                    yield f"{octave+2}_0", pool_type(kernel_size=(2, 2))
                previous = module.filters

    def extract_one(self, image, layer, start):
        names = ["0_0"] + list(self.features._modules.keys())
        index = names.index(layer)

        for i in range(names.index(start) + 1, index + 1):
            image = self.features[i - 1].forward(image)
        return image

    def _get_extractors(self, layers, start):
        layers = ["0_0"] + layers
        for prev, cur in zip(layers[:-1], layers[1:]):
            yield cur, functools.partial(self.extract_one, layer=cur, start=prev)

    def extract(self, img, layers, start="0_0", as_checkpoints=False):
        for layer, func in self._get_extractors(layers, start):
            if as_checkpoints is True:
                img = tcp.checkpoint(func, img)
            else:
                img = func(img)
            yield layer, img
