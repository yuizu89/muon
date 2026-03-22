from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from cli import validate_experiment_args
from data_utils import build_dataloaders
from engine import evaluate, resolve_device, train_one_epoch
from models import VisionTransformer
from optimizers import build_optimizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> VisionTransformer:
    return VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=10,
        dim=args.model_dim,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        embedding_dropout=args.embedding_dropout,
    )


def namespace_to_dict(args: argparse.Namespace) -> Dict[str, object]:
    config = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name

    if args.train_subset_size > 0:
        subset_name = "n{0}".format(args.train_subset_size)
    else:
        subset_name = "ratio{0:g}".format(args.train_subset_ratio)

    return "vit_cifar10_{0}_{1}_seed{2}".format(
        args.optimizer,
        subset_name,
        args.seed,
    )


def prepare_run_dir(args: argparse.Namespace) -> Path:
    run_dir = args.output_dir / build_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def aggregate_results(run_summaries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    optimizer_names = sorted({summary["optimizer"] for summary in run_summaries})
    aggregate_rows: List[Dict[str, object]] = []

    for optimizer_name in optimizer_names:
        best_scores = [
            summary["best_val_accuracy"]
            for summary in run_summaries
            if summary["optimizer"] == optimizer_name
        ]
        final_scores = [
            summary["final_val_accuracy"]
            for summary in run_summaries
            if summary["optimizer"] == optimizer_name
        ]

        aggregate_rows.append(
            {
                "optimizer": optimizer_name,
                "num_runs": len(best_scores),
                "best_val_accuracy_mean": round(statistics.mean(best_scores), 6),
                "best_val_accuracy_std": round(
                    statistics.pstdev(best_scores) if len(best_scores) > 1 else 0.0,
                    6,
                ),
                "final_val_accuracy_mean": round(statistics.mean(final_scores), 6),
                "final_val_accuracy_std": round(
                    statistics.pstdev(final_scores) if len(final_scores) > 1 else 0.0,
                    6,
                ),
            }
        )

    return aggregate_rows


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    validate_experiment_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    run_dir = prepare_run_dir(args)
    write_json(run_dir / "config.json", namespace_to_dict(args))

    dataloaders = build_dataloaders(args, device)
    train_loader = dataloaders["train_loader"]
    eval_loader = dataloaders["eval_loader"]

    model = build_model(args).to(device)
    optimizer = build_optimizer(model, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(len(train_loader) * args.warmup_epochs)
    base_lrs = [parameter_group["lr"] for parameter_group in optimizer.param_groups]

    history: List[Dict[str, object]] = []
    global_step = 0
    best_eval_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
            base_lrs=base_lrs,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        global_step = int(train_metrics.pop("global_step"))

        eval_metrics = evaluate(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_loss": round(eval_metrics["loss"], 6),
            "val_accuracy": round(eval_metrics["accuracy"], 6),
            "lr": round(train_metrics["lr"], 10),
        }
        history.append(history_row)

        if eval_metrics["accuracy"] > best_eval_accuracy:
            best_eval_accuracy = eval_metrics["accuracy"]
            best_epoch = epoch
            if args.save_checkpoint:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": namespace_to_dict(args),
                        "epoch": epoch,
                        "best_val_accuracy": best_eval_accuracy,
                    },
                    run_dir / "best.pt",
                )

        print(
            "Epoch {0:03d}/{1:03d} | train_loss={2:.4f} | train_acc={3:.4f} | "
            "val_loss={4:.4f} | val_acc={5:.4f} | lr={6:.6f}".format(
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                eval_metrics["loss"],
                eval_metrics["accuracy"],
                train_metrics["lr"],
            )
        )

    write_rows(run_dir / "history.csv", history)

    summary = {
        "run_name": run_dir.name,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "train_samples": dataloaders["train_samples"],
        "eval_samples": dataloaders["eval_samples"],
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_val_accuracy": round(best_eval_accuracy, 6),
        "best_epoch": best_epoch,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else 0.0,
        "history_path": str(run_dir / "history.csv"),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def run_comparison(args: argparse.Namespace) -> List[Dict[str, object]]:
    validate_experiment_args(args)

    comparison_dir = args.output_dir / args.comparison_name
    runs_dir = comparison_dir / "runs"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: List[Dict[str, object]] = []

    for optimizer_name in args.optimizers:
        for seed in args.seeds:
            run_args = argparse.Namespace(**vars(args))
            run_args.optimizer = optimizer_name
            run_args.seed = seed
            run_args.output_dir = runs_dir
            run_args.run_name = "{0}_{1}_seed{2}".format(
                args.comparison_name,
                optimizer_name,
                seed,
            )

            summary = run_experiment(run_args)
            run_summaries.append(summary)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate_rows = aggregate_results(run_summaries)
    write_rows(comparison_dir / "runs.csv", run_summaries)
    write_rows(comparison_dir / "summary.csv", aggregate_rows)
    write_json(
        comparison_dir / "comparison.json",
        {
            "comparison_name": args.comparison_name,
            "config": namespace_to_dict(args),
            "runs": run_summaries,
            "summary": aggregate_rows,
        },
    )

    for row in aggregate_rows:
        print(
            "{0}: best={1:.4f}±{2:.4f}, final={3:.4f}±{4:.4f}".format(
                row["optimizer"],
                row["best_val_accuracy_mean"],
                row["best_val_accuracy_std"],
                row["final_val_accuracy_mean"],
                row["final_val_accuracy_std"],
            )
        )

    return aggregate_rows
