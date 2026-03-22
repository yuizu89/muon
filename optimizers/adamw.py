from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch


def _use_weight_decay(parameter_name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim < 2:
        return False
    if parameter_name.endswith("bias"):
        return False
    if "cls_token" in parameter_name or "pos_embedding" in parameter_name:
        return False
    return True


def build_adamw_parameter_groups(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    weight_decay: float,
) -> List[dict]:
    decay_parameters: List[torch.nn.Parameter] = []
    no_decay_parameters: List[torch.nn.Parameter] = []

    for parameter_name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if _use_weight_decay(parameter_name, parameter):
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups: List[dict] = []
    if decay_parameters:
        parameter_groups.append(
            {"params": decay_parameters, "weight_decay": weight_decay}
        )
    if no_decay_parameters:
        parameter_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})
    return parameter_groups


def build_adamw_optimizer(
    model,
    lr: float,
    weight_decay: float,
    betas: Sequence[float],
    eps: float,
) -> torch.optim.Optimizer:
    parameter_groups = build_adamw_parameter_groups(
        model.named_parameters(),
        weight_decay=weight_decay,
    )
    return torch.optim.AdamW(parameter_groups, lr=lr, betas=betas, eps=eps)
