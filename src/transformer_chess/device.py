from __future__ import annotations

import platform

import torch


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if platform.system() == "Darwin" and mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return get_default_device()
