# Color conversions (RGB, XYZ, Lab) based on an implementation by Hannes Perrot.

import torch
import torch.nn.functional as F


def rgb2xyz(img):
    assert len(img.shape) == 4 and img.shape[1] == 3

    img = torch.where(img > 0.04045, ((img + 0.055) / 1.055).pow(2.4), img / 12.92)
    kernel = img.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    ).view(3, 3, 1, 1)
    return F.conv2d(img, kernel)


def xyz2rgb(img):
    assert len(img.shape) == 4 and img.shape[1] == 3

    kernel = torch.tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    ).view(3, 3, 1, 1)

    result = F.conv2d(img, kernel)
    return torch.where(
        result > 0.0031308, 1.055 * (result.pow(1.0 / 2.4)) - 0.055, 12.92 * result
    )


def _lab_f(t):
    return torch.where(
        t > 0.008856451679035631,
        t.pow(1 / 3),
        t * 7.787037037037035 + 0.13793103448275862,
    )


def _lab_inv(t):
    return torch.where(
        t > 0.20689655172413793,
        t.pow(3),
        0.12841854934601665 * (t - 0.13793103448275862),
    )


def xyz2lab(img):
    x, y, z = img.chunk(3, dim=1)

    fy = _lab_f(y / 1.00000)
    l = 1.16 * fy - 0.16
    a = 5.0 * (_lab_f(x / 0.95047) - fy)
    b = 2.0 * (fy - _lab_f(z / 1.08883))
    return torch.cat([l, a, b], dim=1)


def lab2xyz(lab, wref=None):
    wref = [0.95047, 1.00000, 1.08883]

    dim = 1 if len(lab.shape) == 4 else 0
    l, a, b = lab.chunk(3, dim=dim)

    l2 = (l + 0.16) / 1.16
    x = wref[0] * _lab_inv(l2 + a / 5)
    y = wref[1] * _lab_inv(l2)
    z = wref[2] * _lab_inv(l2 - b / 2)
    xyz = torch.cat([x, y, z], dim=dim)

    return xyz


def rgb2lab(img):
    return xyz2lab(rgb2xyz(img))


def lab2rgb(img):
    return xyz2rgb(lab2xyz(img))
