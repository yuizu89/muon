from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_dataloaders(args, device: torch.device) -> Dict[str, object]:
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(
        root=str(args.data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    eval_dataset = datasets.CIFAR10(
        root=str(args.data_dir),
        train=False,
        download=True,
        transform=eval_transform,
    )

    subset_size = len(train_dataset)
    if args.train_subset_size > 0:
        subset_size = min(args.train_subset_size, len(train_dataset))
    elif args.train_subset_ratio < 1.0:
        subset_size = max(1, int(len(train_dataset) * args.train_subset_ratio))

    if subset_size < len(train_dataset):
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(train_dataset), generator=generator)[:subset_size]
        train_dataset = Subset(train_dataset, indices.tolist())

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs
    )

    return {
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }
