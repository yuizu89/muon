from .adamw import build_adamw_optimizer
from .muon import MuonWithAuxAdam, build_muon_optimizer
from .sam import SAM, build_sam_optimizer


def build_optimizer(model, args):
    optimizer_name = args.optimizer.lower()

    if optimizer_name == "adamw":
        return build_adamw_optimizer(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

    if optimizer_name == "sam":
        return build_sam_optimizer(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            rho=args.sam_rho,
            adaptive=args.sam_adaptive,
        )

    if optimizer_name == "muon":
        return build_muon_optimizer(
            model=model,
            muon_lr=args.muon_lr,
            aux_lr=args.muon_aux_lr,
            muon_weight_decay=args.muon_weight_decay,
            aux_weight_decay=args.muon_aux_weight_decay,
            momentum=args.muon_momentum,
            betas=(args.muon_aux_beta1, args.muon_aux_beta2),
            eps=args.muon_aux_eps,
            ns_steps=args.muon_ns_steps,
            nesterov=args.muon_nesterov,
        )

    raise ValueError("Unsupported optimizer: {0}".format(args.optimizer))


def is_sam_optimizer(optimizer) -> bool:
    return isinstance(optimizer, SAM)


__all__ = ["MuonWithAuxAdam", "SAM", "build_optimizer", "is_sam_optimizer"]
