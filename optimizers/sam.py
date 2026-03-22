from __future__ import annotations

from typing import Sequence, Type

import torch

from .adamw import build_adamw_parameter_groups


class SAM(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer: Type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs
    ) -> None:
        if rho <= 0:
            raise ValueError("rho must be positive")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        if grad_norm.item() == 0.0:
            if zero_grad:
                self.zero_grad(set_to_none=True)
            return

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                self.state[parameter]["old_parameter"] = parameter.data.clone()
                if group["adaptive"]:
                    perturbation = parameter.pow(2) * parameter.grad
                else:
                    perturbation = parameter.grad
                perturbation = perturbation * scale.to(parameter)
                parameter.add_(perturbation)

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.data.copy_(self.state[parameter]["old_parameter"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires a closure")

        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def state_dict(self):
        state = super().state_dict()
        state["base_optimizer"] = self.base_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict):
        base_optimizer_state = state_dict.pop("base_optimizer", None)
        super().load_state_dict(state_dict)
        if base_optimizer_state is not None:
            self.base_optimizer.load_state_dict(base_optimizer_state)
        self.param_groups = self.base_optimizer.param_groups

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if group["adaptive"]:
                    grad = parameter.abs() * parameter.grad
                else:
                    grad = parameter.grad
                norms.append(grad.norm(p=2).to(shared_device))

        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


def build_sam_optimizer(
    model,
    lr: float,
    weight_decay: float,
    betas: Sequence[float],
    eps: float,
    rho: float,
    adaptive: bool,
) -> SAM:
    parameter_groups = build_adamw_parameter_groups(
        model.named_parameters(),
        weight_decay=weight_decay,
    )
    return SAM(
        parameter_groups,
        base_optimizer=torch.optim.AdamW,
        lr=lr,
        betas=betas,
        eps=eps,
        rho=rho,
        adaptive=adaptive,
    )
