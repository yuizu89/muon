from __future__ import annotations

import argparse
from pathlib import Path


OPTIMIZER_CHOICES = ["adamw", "sam", "muon"]


def build_common_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--embedding-dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--save-checkpoint", action="store_true")

    parser.add_argument("--train-subset-ratio", type=float, default=1.0)
    parser.add_argument("--train-subset-size", type=int, default=0)

    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)

    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--sam-adaptive", action="store_true")

    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-aux-lr", type=float, default=3e-4)
    parser.add_argument("--muon-weight-decay", type=float, default=0.05)
    parser.add_argument("--muon-aux-weight-decay", type=float, default=0.05)
    parser.add_argument("--muon-aux-beta1", type=float, default=0.9)
    parser.add_argument("--muon-aux-beta2", type=float, default=0.95)
    parser.add_argument("--muon-aux-eps", type=float, default=1e-10)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-nesterov", dest="muon_nesterov", action="store_true")
    parser.add_argument(
        "--no-muon-nesterov",
        dest="muon_nesterov",
        action="store_false",
    )
    parser.set_defaults(muon_nesterov=True)

    return parser


def build_train_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Vision Transformer on CIFAR-10",
        parents=[build_common_argument_parser()],
    )
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default="adamw",
    )
    return parser


def build_compare_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare optimizers for ViT on CIFAR-10",
        parents=[build_common_argument_parser()],
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=OPTIMIZER_CHOICES,
        default=list(OPTIMIZER_CHOICES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--comparison-name", type=str, default="vit_cifar10_comparison")
    return parser


def build_plot_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot training curves from a run or comparison directory",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a run directory or a comparison directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots. Defaults to <input_path>/plots",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Image format used for saved plots",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def validate_experiment_args(args: argparse.Namespace) -> None:
    if args.train_subset_ratio <= 0.0 or args.train_subset_ratio > 1.0:
        raise ValueError("--train-subset-ratio must be within (0, 1]")
    if args.train_subset_size < 0:
        raise ValueError("--train-subset-size must be non-negative")
    if args.image_size != 32:
        raise ValueError("This experiment currently expects CIFAR-10 image size 32")


def validate_plot_args(args: argparse.Namespace) -> None:
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
