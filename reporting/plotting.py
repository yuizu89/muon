from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass
class RunHistory:
    run_dir: Path
    optimizer: str
    history: List[Dict[str, float]]


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        return {}
    return payload


def _load_history(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = {
                "epoch": float(raw_row["epoch"]),
                "train_loss": float(raw_row["train_loss"]),
                "train_accuracy": float(raw_row["train_accuracy"]),
                "val_loss": float(raw_row["val_loss"]),
                "val_accuracy": float(raw_row["val_accuracy"]),
                "lr": float(raw_row["lr"]),
            }
            rows.append(row)

    if not rows:
        raise ValueError("History file is empty: {0}".format(path))
    return rows


def _resolve_optimizer_name(run_dir: Path) -> str:
    summary = _load_json(run_dir / "summary.json")
    optimizer = summary.get("optimizer")
    if isinstance(optimizer, str) and optimizer:
        return optimizer

    config = _load_json(run_dir / "config.json")
    optimizer = config.get("optimizer")
    if isinstance(optimizer, str) and optimizer:
        return optimizer

    return run_dir.name


def _load_run_history(run_dir: Path) -> RunHistory:
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError("Missing history.csv in {0}".format(run_dir))

    return RunHistory(
        run_dir=run_dir,
        optimizer=_resolve_optimizer_name(run_dir),
        history=_load_history(history_path),
    )


def _get_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install requirements.txt first."
        ) from exc

    return plt


def _build_output_dir(input_path: Path, output_dir: Path | None) -> Path:
    target_dir = output_dir if output_dir is not None else input_path / "plots"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _plot_single_run(
    run_history: RunHistory,
    output_dir: Path,
    file_format: str,
    dpi: int,
) -> Path:
    plt = _get_pyplot()
    epochs = [int(row["epoch"]) for row in run_history.history]

    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in run_history.history],
        label="train",
        linewidth=2.0,
    )
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in run_history.history],
        label="val",
        linewidth=2.0,
    )
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in run_history.history],
        label="train",
        linewidth=2.0,
    )
    axes[1].plot(
        epochs,
        [row["val_accuracy"] for row in run_history.history],
        label="val",
        linewidth=2.0,
    )
    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].plot(
        epochs,
        [row["lr"] for row in run_history.history],
        color="tab:green",
        linewidth=2.0,
    )
    axes[2].set_title("Learning Rate")
    axes[2].set_ylabel("LR")
    axes[2].set_xlabel("Epoch")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for axis in axes:
        axis.grid(True, alpha=0.3)

    figure.suptitle(
        "{0} ({1})".format(run_history.run_dir.name, run_history.optimizer),
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    output_path = output_dir / "{0}_training_curves.{1}".format(
        run_history.run_dir.name,
        file_format,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _aggregate_metric(
    run_histories: Sequence[RunHistory],
    metric_name: str,
) -> Tuple[List[int], List[float], List[float]]:
    values_by_epoch: Dict[int, List[float]] = {}

    for run_history in run_histories:
        for row in run_history.history:
            epoch = int(row["epoch"])
            values_by_epoch.setdefault(epoch, []).append(row[metric_name])

    epochs = sorted(values_by_epoch)
    means = [statistics.mean(values_by_epoch[epoch]) for epoch in epochs]
    stds = [
        statistics.pstdev(values_by_epoch[epoch])
        if len(values_by_epoch[epoch]) > 1
        else 0.0
        for epoch in epochs
    ]
    return epochs, means, stds


def _plot_comparison(
    comparison_dir: Path,
    run_histories: Sequence[RunHistory],
    output_dir: Path,
    file_format: str,
    dpi: int,
) -> Path:
    plt = _get_pyplot()
    grouped_histories: Dict[str, List[RunHistory]] = {}
    for run_history in run_histories:
        grouped_histories.setdefault(run_history.optimizer, []).append(run_history)

    metric_specs = [
        ("train_loss", "Train Loss", "Loss"),
        ("val_loss", "Validation Loss", "Loss"),
        ("train_accuracy", "Train Accuracy", "Accuracy"),
        ("val_accuracy", "Validation Accuracy", "Accuracy"),
        ("lr", "Learning Rate", "LR"),
    ]

    figure, axes = plt.subplots(3, 2, figsize=(14, 14))
    flat_axes = list(axes.flatten())
    color_map = plt.get_cmap("tab10")

    for optimizer_index, optimizer_name in enumerate(sorted(grouped_histories)):
        color = color_map(optimizer_index % 10)
        optimizer_histories = grouped_histories[optimizer_name]

        for axis, (metric_name, title, ylabel) in zip(flat_axes, metric_specs):
            epochs, means, stds = _aggregate_metric(optimizer_histories, metric_name)
            lower = [mean - std for mean, std in zip(means, stds)]
            upper = [mean + std for mean, std in zip(means, stds)]

            axis.plot(
                epochs,
                means,
                label=optimizer_name,
                color=color,
                linewidth=2.0,
            )
            axis.fill_between(epochs, lower, upper, color=color, alpha=0.15)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)

            if metric_name == "lr":
                axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    flat_axes[-1].axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))

    figure.suptitle(
        "{0} (mean ± std across runs)".format(comparison_dir.name),
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output_path = output_dir / "{0}_comparison_curves.{1}".format(
        comparison_dir.name,
        file_format,
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_results(
    input_path: Path,
    output_dir: Path | None = None,
    file_format: str = "png",
    dpi: int = 150,
) -> List[Path]:
    resolved_input = input_path.expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError("Path does not exist: {0}".format(input_path))

    resolved_output = (
        output_dir.expanduser().resolve() if output_dir is not None else None
    )
    target_output_dir = _build_output_dir(resolved_input, resolved_output)

    if (resolved_input / "history.csv").exists():
        run_history = _load_run_history(resolved_input)
        return [
            _plot_single_run(
                run_history=run_history,
                output_dir=target_output_dir,
                file_format=file_format,
                dpi=dpi,
            )
        ]

    runs_dir = resolved_input / "runs"
    if runs_dir.is_dir():
        run_dirs = sorted(
            run_dir
            for run_dir in runs_dir.iterdir()
            if run_dir.is_dir() and (run_dir / "history.csv").exists()
        )
        if not run_dirs:
            raise FileNotFoundError(
                "No run directories with history.csv were found in {0}".format(runs_dir)
            )

        run_histories = [_load_run_history(run_dir) for run_dir in run_dirs]
        return [
            _plot_comparison(
                comparison_dir=resolved_input,
                run_histories=run_histories,
                output_dir=target_output_dir,
                file_format=file_format,
                dpi=dpi,
            )
        ]

    raise ValueError(
        "Expected a run directory with history.csv or a comparison directory with runs/"
    )
