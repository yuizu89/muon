from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from optimizers import is_sam_optimizer


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    if total_steps <= warmup_steps:
        return 1.0

    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_learning_rate_schedule(
    optimizer: torch.optim.Optimizer,
    base_lrs: List[float],
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> List[float]:
    multiplier = lr_multiplier(step, total_steps, warmup_steps)
    current_lrs = []

    for parameter_group, base_lr in zip(optimizer.param_groups, base_lrs):
        parameter_group["lr"] = base_lr * multiplier
        current_lrs.append(parameter_group["lr"])

    return current_lrs


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return predictions.eq(targets).float().sum().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    base_lrs: List[float],
    global_step: int,
    total_steps: int,
    warmup_steps: int,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    total_correct = 0.0
    total_examples = 0
    current_lrs = list(base_lrs)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        batch_size = targets.size(0)

        current_lrs = apply_learning_rate_schedule(
            optimizer=optimizer,
            base_lrs=base_lrs,
            step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

        optimizer.zero_grad(set_to_none=True)

        if is_sam_optimizer(optimizer):
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.first_step(zero_grad=True)

            sam_logits = model(inputs)
            sam_loss = criterion(sam_logits, targets)
            sam_loss.backward()

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.second_step(zero_grad=True)
            loss_value = sam_loss.item()
            total_correct += accuracy_from_logits(sam_logits.detach(), targets)
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()
            loss_value = loss.item()
            total_correct += accuracy_from_logits(logits.detach(), targets)

        loss_meter.update(loss_value, batch_size)
        total_examples += batch_size
        global_step += 1

    return {
        "loss": loss_meter.average,
        "accuracy": total_correct / max(1, total_examples),
        "lr": sum(current_lrs) / max(1, len(current_lrs)),
        "global_step": global_step,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    total_correct = 0.0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        total_correct += accuracy_from_logits(logits, targets)
        total_examples += batch_size

    return {
        "loss": loss_meter.average,
        "accuracy": total_correct / max(1, total_examples),
    }
