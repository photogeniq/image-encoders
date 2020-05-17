# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import os
import bz2
import urllib
import hashlib
import progressbar


DATA_URL = "https://github.com/Photogeniq/image-encoders/releases/download/"
DATA_DIR = os.path.expanduser(f"~/.cache/torch")
DATA_CHUNK = 8192


def download_to_file(model, hexdigest):
    fullpath = f"{DATA_DIR}/{model}.pkl"
    if os.path.exists(fullpath):
        return fullpath

    directory, filename = os.path.split(fullpath)
    os.makedirs(directory, exist_ok=True)
    response = urllib.request.urlopen(f"{DATA_URL}/{model}.pkl.bz2")

    widgets = [
        progressbar.Percentage(),
        progressbar.Bar(marker="■", fill="·"),
        progressbar.DataSize(),
        " ",
        progressbar.ETA(),
    ]
    bunzip, output, hasher = (
        bz2.BZ2Decompressor(),
        open(fullpath, "wb"),
        hashlib.new("md5"),
    )

    with progressbar.ProgressBar(max_value=response.length, widgets=widgets) as bar:
        for i in range((response.length // DATA_CHUNK) + 1):
            chunk = response.read(DATA_CHUNK)
            data = bunzip.decompress(chunk)

            bar.update(i * DATA_CHUNK)
            hasher.update(data)
            output.write(data)

    assert (
        hasher.hexdigest() == hexdigest
    ), "ERROR: `{model}` has unexpected MD5 checksum."
    return fullpath
