from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .device import get_default_device, get_model_device


TRAINING_PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")
_AMP_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def make_optimizer(model: nn.Module, *, lr: float = 3e-4, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _resolve_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    explicit_device = device is not None
    target_device = torch.device(device) if explicit_device else get_model_device(model)
    if not explicit_device and target_device.type == "cpu":
        target_device = get_default_device()
    return target_device


def _cuda_bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is not None:
        return bool(checker())
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def resolve_training_precision(device: torch.device | str, precision: str = "auto") -> str:
    target_device = torch.device(device)
    normalized = precision.lower()
    if normalized not in TRAINING_PRECISION_CHOICES:
        raise ValueError(f"Unsupported precision {precision!r}. Expected one of {TRAINING_PRECISION_CHOICES}.")

    if target_device.type != "cuda":
        if normalized in {"auto", "fp32"}:
            return "fp32"
        raise ValueError(f"Precision {normalized!r} requires CUDA, but got device {target_device.type!r}.")

    if normalized == "auto":
        return "bf16" if _cuda_bf16_supported(target_device) else "fp16"
    if normalized == "bf16" and not _cuda_bf16_supported(target_device):
        raise ValueError("bf16 was requested, but the selected CUDA device does not support bf16.")
    return normalized


def _autocast_kwargs(device: torch.device, precision: str) -> dict[str, object]:
    resolved_precision = resolve_training_precision(device, precision)
    kwargs: dict[str, object] = {
        "device_type": device.type,
        "enabled": resolved_precision in _AMP_DTYPES,
    }
    amp_dtype = _AMP_DTYPES.get(resolved_precision)
    if amp_dtype is not None:
        kwargs["dtype"] = amp_dtype
    return kwargs


def _grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler | None:
    resolved_precision = resolve_training_precision(device, precision)
    if device.type == "cuda" and resolved_precision == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    precision: str = "auto",
    scaler: torch.amp.GradScaler | None = None,
    loss_fn: nn.Module | None = None,
) -> float:
    if inputs.ndim != 4:
        raise ValueError("inputs must have shape (batch, channels, 8, 8).")
    if targets.ndim != 1:
        raise ValueError("targets must have shape (batch,).")

    criterion = loss_fn or nn.MSELoss()
    target_device = _resolve_device(model, device)
    non_blocking = target_device.type == "cuda"
    autocast_kwargs = _autocast_kwargs(target_device, precision)
    active_scaler = scaler if scaler is not None else _grad_scaler(target_device, precision)
    model.to(target_device)
    inputs = inputs.to(target_device, non_blocking=non_blocking)
    targets = targets.to(target_device, non_blocking=non_blocking)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(**autocast_kwargs):
        predictions = model(inputs)
        loss = criterion(predictions.float(), targets.float())

    if active_scaler is not None:
        active_scaler.scale(loss).backward()
        active_scaler.step(optimizer)
        active_scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str | None = None,
    precision: str = "auto",
    max_batches: int | None = None,
    loss_fn: nn.Module | None = None,
) -> float:
    criterion = loss_fn or nn.MSELoss()
    target_device = _resolve_device(model, device)
    non_blocking = target_device.type == "cuda"
    autocast_kwargs = _autocast_kwargs(target_device, precision)

    model.to(target_device)
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch_index, (inputs, targets) in enumerate(batches, start=1):
        inputs = inputs.to(target_device, non_blocking=non_blocking)
        targets = targets.to(target_device, non_blocking=non_blocking)
        with torch.autocast(**autocast_kwargs):
            predictions = model(inputs)
            total_loss += float(criterion(predictions.float(), targets.float()).item())
        total_batches += 1
        if max_batches is not None and batch_index >= max_batches:
            break

    if total_batches == 0:
        raise ValueError("No batches were provided for evaluation.")
    return total_loss / total_batches


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str | None = None,
    precision: str = "auto",
    max_batches: int | None = None,
    loss_fn: nn.Module | None = None,
) -> float:
    target_device = _resolve_device(model, device)
    scaler = _grad_scaler(target_device, precision)
    model.to(target_device)
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch_index, (inputs, targets) in enumerate(batches, start=1):
        total_loss += train_step(
            model,
            optimizer,
            inputs,
            targets,
            device=target_device,
            precision=precision,
            scaler=scaler,
            loss_fn=loss_fn,
        )
        total_batches += 1
        if max_batches is not None and batch_index >= max_batches:
            break

    if total_batches == 0:
        raise ValueError("No batches were provided for training.")
    return total_loss / total_batches
