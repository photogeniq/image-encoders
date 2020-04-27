# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import torch.nn

from base import Encoder, ModuleConfig as M


def ConvBlock(in_channels, out_channels, activation="ReLU"):
    layers = [
        torch.nn.ReflectionPad2d(1),
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)),
        getattr(torch.nn, activation)(inplace=True),
    ]
    return torch.nn.Sequential(*layers)


class VGG19(Encoder):
    CONFIG = [M(f=64, b=2), M(f=128, b=2), M(f=256, b=4), M(f=512, b=4), M(f=512, b=4)]

    def __init__(self, pretrained=True, **kwargs):
        blocks = self.make_blocks(self.CONFIG, ConvBlock)
        super(VGG19, self).__init__(blocks, **kwargs)

        if pretrained is True:
            self.load_state_dict(torch.load("models/vgg19.pkl"), strict=True)


class VGG16(Encoder):
    CONFIG = [M(f=64, b=2), M(f=128, b=2), M(f=256, b=3), M(f=512, b=3), M(f=512, b=3)]

    def __init__(self, pretrained=True, **kwargs):
        blocks = self.make_blocks(self.CONFIG, ConvBlock)
        super(VGG16, self).__init__(blocks, **kwargs)

        if pretrained is True:
            self.load_state_dict(torch.load("models/vgg16.pkl"), strict=True)


class VGG13(Encoder):
    CONFIG = [M(f=64, b=2), M(f=128, b=2), M(f=256, b=2), M(f=512, b=2), M(f=512, b=2)]

    def __init__(self, pretrained=True, **kwargs):
        blocks = self.make_blocks(self.CONFIG, ConvBlock)
        super(VGG13, self).__init__(blocks, **kwargs)

        if pretrained is True:
            self.load_state_dict(torch.load("models/vgg13.pkl"), strict=True)


class VGG11(Encoder):
    CONFIG = [M(f=64, b=1), M(f=128, b=1), M(f=256, b=2), M(f=512, b=2), M(f=512, b=2)]

    def __init__(self, pretrained=True, **kwargs):
        blocks = self.make_blocks(self.CONFIG, ConvBlock)
        super(VGG11, self).__init__(blocks, **kwargs)

        if pretrained is True:
            self.load_state_dict(torch.load("models/vgg11.pkl"), strict=True)


class ThinetSmall(Encoder):
    CONFIG = [M(f=32, b=2), M(f=64, b=2), M(f=128, b=3), M(f=256, b=3), M(f=512, b=3)]

    def __init__(self, pretrained=True, **kwargs):
        blocks = self.make_blocks(self.CONFIG, ConvBlock)
        super(ThinetSmall, self).__init__(blocks, **kwargs)

        if pretrained is True:
            self.load_state_dict(torch.load("models/thinet_small.pkl"), strict=True)


class ThinetTiny(Encoder):
    CONFIG = [M(f=16, b=2), M(f=32, b=2), M(f=64, b=3), M(f=128, b=3), M(f=128, b=2)]

    def __init__(self, pretrained=True, **kwargs):
        blocks = self.make_blocks(self.CONFIG, ConvBlock)
        super(ThinetTiny, self).__init__(blocks, **kwargs)

        if pretrained is True:
            self.load_state_dict(torch.load("models/thinet_tiny.pkl"), strict=True)
