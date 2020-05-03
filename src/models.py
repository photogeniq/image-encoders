# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

from torch import nn

from base import Encoder, ModuleConfig as M


def ConvBlock(in_channels, out_channels, activation="ReLU"):
    layers = [
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)),
        getattr(nn, activation)(inplace=True),
    ]
    return nn.Sequential(*layers)


class VGG19(Encoder):
    CONFIG = [M(f=64, b=2), M(f=128, b=2), M(f=256, b=4), M(f=512, b=4), M(f=512, b=4)]

    def __init__(self, block_type=ConvBlock, pool_type=nn.MaxPool2d, pretrained=True, **kwargs):
        super(VGG19, self).__init__(self.CONFIG, block_type, pool_type, input_type="RGB", **kwargs)

        if pretrained is True:
            self.load_pretrained("v0.1/vgg19", hexdigest="fd09b2e521af4375dc5e2a47cbe4da75")


class VGG16(Encoder):
    CONFIG = [M(f=64, b=2), M(f=128, b=2), M(f=256, b=3), M(f=512, b=3), M(f=512, b=3)]

    def __init__(self, block_type=ConvBlock, pool_type=nn.MaxPool2d, pretrained=True, **kwargs):
        super(VGG16, self).__init__(self.CONFIG, block_type, pool_type, input_type="RGB", **kwargs)

        if pretrained is True:
            self.load_pretrained("v0.1/vgg16", hexdigest="1be9714ee6c3508820ee6d0758aaeb01")


class VGG13(Encoder):
    CONFIG = [M(f=64, b=2), M(f=128, b=2), M(f=256, b=2), M(f=512, b=2), M(f=512, b=2)]

    def __init__(self, block_type=ConvBlock, pool_type=nn.MaxPool2d, pretrained=True, **kwargs):
        super(VGG13, self).__init__(self.CONFIG, block_type, pool_type, input_type="RGB", **kwargs)

        if pretrained is True:
            self.load_pretrained("v0.1/vgg13", hexdigest="4aef6aa089421b12097c947f2f8257b4")


class VGG11(Encoder):
    CONFIG = [M(f=64, b=1), M(f=128, b=1), M(f=256, b=2), M(f=512, b=2), M(f=512, b=2)]

    def __init__(self, block_type=ConvBlock, pool_type=nn.MaxPool2d, pretrained=True, **kwargs):
        super(VGG11, self).__init__(self.CONFIG, block_type, pool_type, input_type="RGB", **kwargs)

        if pretrained is True:
            self.load_pretrained("v0.1/vgg11", hexdigest="28632f54ab04986aa0f530034f5a4f96")


class ThinetSmall(Encoder):
    CONFIG = [M(f=32, b=2), M(f=64, b=2), M(f=128, b=3), M(f=256, b=3), M(f=512, b=3)]

    def __init__(self, block_type=ConvBlock, pool_type=nn.MaxPool2d, pretrained=True, **kwargs):
        super(VGG11, self).__init__(self.CONFIG, block_type, pool_type, input_type="RGB", **kwargs)

        if pretrained is True:
            self.load_pretrained("v0.1/thinet_small", hexdigest="bbd2c25d2e8f77ffee09d1e2d460574f")


class ThinetTiny(Encoder):
    CONFIG = [M(f=16, b=2), M(f=32, b=2), M(f=64, b=3), M(f=128, b=3), M(f=128, b=2)]

    def __init__(self, block_type=ConvBlock, pool_type=nn.MaxPool2d, pretrained=True, **kwargs):
        super(VGG11, self).__init__(self.CONFIG, block_type, pool_type, input_type="RGB", **kwargs)

        if pretrained is True:
            self.load_pretrained("v0.1/thinet_tiny", hexdigest="0bf93a6387f45ac1e15f3483d8d71dcc")
