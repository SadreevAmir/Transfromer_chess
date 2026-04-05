from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .device import get_default_device, get_model_device


def make_optimizer(model: nn.Module, *, lr: float = 3e-4, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _resolve_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    explicit_device = device is not None
    target_device = torch.device(device) if explicit_device else get_model_device(model)
    if not explicit_device and target_device.type == "cpu":
        target_device = get_default_device()
    return target_device


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    loss_fn: nn.Module | None = None,
) -> float:
    if inputs.ndim != 4:
        raise ValueError("inputs must have shape (batch, channels, 8, 8).")
    if targets.ndim != 1:
        raise ValueError("targets must have shape (batch,).")

    criterion = loss_fn or nn.MSELoss()
    target_device = _resolve_device(model, device)
    non_blocking = target_device.type == "cuda"
    inputs = inputs.to(target_device, non_blocking=non_blocking)
    targets = targets.to(target_device, non_blocking=non_blocking)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
    loss_fn: nn.Module | None = None,
) -> float:
    criterion = loss_fn or nn.MSELoss()
    target_device = _resolve_device(model, device)
    non_blocking = target_device.type == "cuda"

    model.to(target_device)
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch_index, (inputs, targets) in enumerate(batches, start=1):
        inputs = inputs.to(target_device, non_blocking=non_blocking)
        targets = targets.to(target_device, non_blocking=non_blocking)
        predictions = model(inputs)
        total_loss += float(criterion(predictions, targets).item())
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
    max_batches: int | None = None,
    loss_fn: nn.Module | None = None,
) -> float:
    target_device = _resolve_device(model, device)
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
            loss_fn=loss_fn,
        )
        total_batches += 1
        if max_batches is not None and batch_index >= max_batches:
            break

    if total_batches == 0:
        raise ValueError("No batches were provided for training.")
    return total_loss / total_batches
