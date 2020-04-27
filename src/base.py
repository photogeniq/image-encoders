# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import collections

import torch


class ModuleConfig:
    def __init__(self, f, b):
        self.filters = f
        self.blocks = b


class Encoder(torch.nn.Module):
    def __init__(self, blocks, **kwargs):
        super(Encoder, self).__init__()

        self.features = torch.nn.Sequential(collections.OrderedDict(blocks))

    def make_blocks(self, config, block_type):
        previous = 3
        for octave, module in enumerate(config):
            for block in range(module.blocks):
                yield f"{octave+1}_{block+1}", block_type(previous, module.filters)
                if block == module.blocks - 1:
                    yield f"{octave+1}_p", torch.nn.MaxPool2d(kernel_size=(2, 2))
                previous = module.filters

    def extract(self, image, layers: list, start="0_0"):
        if len(layers) == 0:
            raise ValueError("Expecting output layers, e.g. ['1_1', '2_1'].")

        names = ["0_0"] + list(self.features._modules.keys())
        indices = [names.index(l) for l in layers]

        for i in range(names.index(start) + 1, max(indices) + 1):
            image = self.features[i - 1].forward(image)
            if i in indices:
                yield names[i], image
