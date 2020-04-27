# image-encoders — Copyright (c) 2020, Novelty Factory KG.  See LICENSE for details.

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--device", action="store", default="cpu", help="my option: cpu or cuda"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as using gpu only")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--device") == "cuda":
        return

    skip_cuda = pytest.mark.skip(reason="need cuda device to run tests")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
