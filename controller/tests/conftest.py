"""Shared pytest setup. Lives inside the package so it moves with it on merge
(the shared repo needs no pytest config of its own)."""

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "m1: needs M1's environment, skipped when it can't be imported")


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(params=_available_devices())
def device(request):
    """A test that takes `device` runs once per available device (cpu, plus cuda / mps)."""
    return torch.device(request.param)
