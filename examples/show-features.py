# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.
"""
Usage:
    show-features [--resolution=<res>] [--model=<mdl>] [--layers=<layers>] FILES ...
    show-features --help

Options:
    FILES               Glob pattern which images to load.
    --resolution=<res>  Resize the images before processing them. [default: 256x256]
    --model=<mdl>       The encoder to use to extract features. [default: VGG11]
    --layers=<layers>   Comma separated list of layers to extract. [default: 1_1,2_1,3_1,4_1]
    -h --help           Show this message.
"""

# Built-in Python modules.
import os
import glob
import itertools

# Third-party libraries.
import docopt
import PIL.Image
import matplotlib.pyplot as plt

# Deep Learning framework.
import torch
import torch.nn.functional as F
import torchvision.transforms

# This library!
from encoders import models


def gram_matrix(features):
    """Commute the pixelwise correlation in a feature map using matrix multiplication.

    Parameters
    ----------
        features : Tensor (b, ch, h, w)

    Result
    ------
        gram : Tensor (b, ch, ch)
    """
    (b, ch, h, w) = features.size()
    f_i = features.view(b, ch, w * h)
    f_t = f_i.transpose(1, 2)
    return f_i.bmm(f_t) / (ch * h * w)


def show_features(model, array, filename, layers):
    """Display a window with the gram matrices for the specified array of pixels.
    """
    # Setup matplotlib to draw multiple images in the same figure.
    fig = plt.figure(figsize=(2 * len(layers), 4))
    fig.suptitle(os.path.split(filename)[1], fontsize=16)

    # Iterate over extracted features one by one.
    for i, (layer, features) in enumerate(model.extract(array, layers)):
        # Compute the gram matrix itself.
        gram = gram_matrix(features)

        # Setup this cell of the figure.
        ax = fig.add_subplot(1, len(layers), i + 1)
        ax.set_title(f"L{layer}: {gram.shape[1]}x{gram.shape[2]}")
        ax.set_axis_off()
        ax.autoscale(False)

        # Alignments and normalization.
        lower = gram.mean() - gram.std()
        upper = gram.mean() + gram.std()
        s = 0.25 - 0.25 * (features.shape[1] / 512.0)
        extent = (0.0 + s, 1.0 - s, 0.0 + s, 1.0 - s)

        # Actually show the gram matrix as image.  Use `imsave` to write to disk.
        plt.imshow(gram[0], extent=extent, vmin=lower, vmax=upper)

    plt.show()


def main(config):
    """Load all the files specified by the user and convert to tensors.
    """

    model = getattr(models, config["--model"])(pretrained=True)
    resolution = tuple(map(int, config["--resolution"].split("x")))

    for filename in itertools.chain.from_iterable(
        glob.glob(f) for f in config["FILES"]
    ):
        print(filename)
        img = PIL.Image.open(filename).convert(mode="RGB")
        img = PIL.ImageOps.fit(img, resolution)
        array = torchvision.transforms.ToTensor()(img)
        show_features(model, array, filename, config["--layers"].split(","))


if __name__ == "__main__":
    config = docopt.docopt(__doc__, version="0.2")

    with torch.no_grad():
        main(config)
