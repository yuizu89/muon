from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch


def _zeropower_via_newton_schulz5(
    gradient_matrix: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    if gradient_matrix.ndim < 2:
        raise ValueError("Muon expects at least a 2D tensor")

    a, b, c = (3.4445, -4.7750, 2.0315)
    original_dtype = gradient_matrix.dtype
    matrix = gradient_matrix
    transpose = matrix.size(-2) > matrix.size(-1)

    if transpose:
        matrix = matrix.transpose(-2, -1)

    if matrix.is_cuda:
        matrix = matrix.to(dtype=torch.bfloat16)
    else:
        matrix = matrix.float()

    matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        gram = matrix @ matrix.transpose(-2, -1)
        polynomial = b * gram + c * gram @ gram
        matrix = a * matrix + polynomial @ matrix

    if transpose:
        matrix = matrix.transpose(-2, -1)

    return matrix.to(dtype=original_dtype)


def _muon_update(
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float,
    ns_steps: int,
    nesterov: bool,
) -> torch.Tensor:
    momentum_buffer.lerp_(gradient, 1.0 - beta)
    update = gradient.lerp(momentum_buffer, beta) if nesterov else momentum_buffer

    original_shape = update.shape
    if update.ndim > 2:
        update = update.reshape(update.size(0), -1)

    update = _zeropower_via_newton_schulz5(update, steps=ns_steps)
    update = update * math.sqrt(max(1.0, update.size(-2) / update.size(-1)))
    return update.reshape(original_shape)


def _adam_update(
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: Sequence[float],
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(gradient, 1.0 - betas[0])
    exp_avg_sq.lerp_(gradient.square(), 1.0 - betas[1])

    bias_corrected_avg = exp_avg / (1.0 - betas[0] ** step)
    bias_corrected_avg_sq = exp_avg_sq / (1.0 - betas[1] ** step)
    return bias_corrected_avg / (bias_corrected_avg_sq.sqrt() + eps)


def _use_weight_decay(parameter_name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim < 2:
        return False
    if parameter_name.endswith("bias"):
        return False
    if "cls_token" in parameter_name or "pos_embedding" in parameter_name:
        return False
    return True


class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, parameter_groups: List[dict]) -> None:
        normalized_groups = []

        for group in parameter_groups:
            if "use_muon" not in group:
                raise ValueError("Each param group must include use_muon")

            if group["use_muon"]:
                normalized_groups.append(
                    {
                        "params": list(group["params"]),
                        "use_muon": True,
                        "lr": group.get("lr", 0.02),
                        "momentum": group.get("momentum", 0.95),
                        "weight_decay": group.get("weight_decay", 0.0),
                        "ns_steps": group.get("ns_steps", 5),
                        "nesterov": group.get("nesterov", True),
                    }
                )
            else:
                normalized_groups.append(
                    {
                        "params": list(group["params"]),
                        "use_muon": False,
                        "lr": group.get("lr", 3e-4),
                        "betas": group.get("betas", (0.9, 0.95)),
                        "eps": group.get("eps", 1e-10),
                        "weight_decay": group.get("weight_decay", 0.0),
                    }
                )

        super().__init__(normalized_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue

                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)

                    update = _muon_update(
                        gradient=parameter.grad,
                        momentum_buffer=state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
            else:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue

                    state = self.state[parameter]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0

                    state["step"] += 1
                    update = _adam_update(
                        gradient=parameter.grad,
                        exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"],
                        step=state["step"],
                        betas=group["betas"],
                        eps=group["eps"],
                    )
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])

        return loss


def _named_parameters_with_prefix(module, prefix: str) -> List[Tuple[str, torch.nn.Parameter]]:
    return [
        ("{0}.{1}".format(prefix, name), parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]


def build_muon_optimizer(
    model,
    muon_lr: float,
    aux_lr: float,
    muon_weight_decay: float,
    aux_weight_decay: float,
    momentum: float,
    betas: Sequence[float],
    eps: float,
    ns_steps: int,
    nesterov: bool,
) -> MuonWithAuxAdam:
    hidden_matrix_parameters = [
        parameter
        for parameter in model.body.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    if not hidden_matrix_parameters:
        raise ValueError("Muon requires at least one 2D parameter in model.body")

    auxiliary_named_parameters: List[Tuple[str, torch.nn.Parameter]] = []
    auxiliary_named_parameters.extend(_named_parameters_with_prefix(model.embed, "embed"))
    auxiliary_named_parameters.extend(
        [
            ("body.{0}".format(name), parameter)
            for name, parameter in model.body.named_parameters()
            if parameter.requires_grad and parameter.ndim < 2
        ]
    )
    auxiliary_named_parameters.extend(_named_parameters_with_prefix(model.head, "head"))

    aux_decay_parameters: List[torch.nn.Parameter] = []
    aux_no_decay_parameters: List[torch.nn.Parameter] = []
    for parameter_name, parameter in auxiliary_named_parameters:
        if _use_weight_decay(parameter_name, parameter):
            aux_decay_parameters.append(parameter)
        else:
            aux_no_decay_parameters.append(parameter)

    parameter_groups = [
        {
            "params": hidden_matrix_parameters,
            "use_muon": True,
            "lr": muon_lr,
            "momentum": momentum,
            "weight_decay": muon_weight_decay,
            "ns_steps": ns_steps,
            "nesterov": nesterov,
        }
    ]
    if aux_decay_parameters:
        parameter_groups.append(
            {
                "params": aux_decay_parameters,
                "use_muon": False,
                "lr": aux_lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": aux_weight_decay,
            }
        )
    if aux_no_decay_parameters:
        parameter_groups.append(
            {
                "params": aux_no_decay_parameters,
                "use_muon": False,
                "lr": aux_lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": 0.0,
            }
        )

    return MuonWithAuxAdam(parameter_groups)
