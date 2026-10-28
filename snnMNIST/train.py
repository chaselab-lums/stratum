#!/usr/bin/env python3
# Copyright 2026 N Vision Systems And Technologies SL
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Train the N-MNIST SNN and save weights.pt.

Code generation and the drawn-image test live in run.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import tonic
import tonic.transforms as transforms
import torchvision
from snntorch import functional as SF
from snntorch import surrogate
from snntorch import utils
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset


HERE = Path(__file__).parent


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(
    data_dir: Path,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    time_window_us: int,
    denoise_filter_time_us: int,
) -> tuple[DataLoader, DataLoader]:
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = cast(Any, transforms.Compose)(
        [
            transforms.Denoise(filter_time=denoise_filter_time_us),
            transforms.ToFrame(sensor_size=sensor_size, time_window=time_window_us),
        ]
    )

    trainset = tonic.datasets.NMNIST(save_to=str(data_dir), train=True, transform=frame_transform)
    testset = tonic.datasets.NMNIST(save_to=str(data_dir), train=False, transform=frame_transform)

    train_sample_transform = cast(Any, transforms.Compose)(
        [
            torch.from_numpy,
            torchvision.transforms.RandomRotation(degrees=(-10, 10)),
        ]
    )
    test_sample_transform = torch.from_numpy

    cached_trainset = cast(Any, DiskCachedDataset)(
        trainset,
        cache_path=str(cache_dir / "train"),
        transform=train_sample_transform,
    )
    cached_testset = cast(Any, DiskCachedDataset)(
        testset,
        cache_path=str(cache_dir / "test"),
        transform=test_sample_transform,
    )

    trainloader = DataLoader(
        cast(Any, cached_trainset),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = DataLoader(
        cast(Any, cached_testset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return trainloader, testloader


def build_csnn_network(beta: float) -> nn.Module:
    spike_grad = surrogate.atan()
    return nn.Sequential(
        nn.Conv2d(2, 12, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Conv2d(12, 32, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 10),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    )


def build_mlir_network(input_size: int, hidden_size: int, beta: float) -> nn.Module:
    spike_grad = surrogate.atan()
    return nn.Sequential(
        nn.Linear(input_size, hidden_size, bias=False),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(hidden_size, 10, bias=True),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    )


def forward_pass(net: nn.Module, data_t_bchw: torch.Tensor, flatten_input: bool) -> torch.Tensor:
    utils.reset(net)
    spikes: list[torch.Tensor] = []
    for step in range(data_t_bchw.size(0)):
        x_t = data_t_bchw[step]
        if flatten_input:
            x_t = x_t.reshape(x_t.shape[0], -1)
        spk_out, _mem_out = net(x_t)
        spikes.append(spk_out)
    return torch.stack(spikes, dim=0)


@torch.no_grad()
def evaluate(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    flatten_input: bool,
) -> float:
    net.eval()
    total = 0.0
    count = 0

    for i, (data, targets) in enumerate(loader):
        if i >= max_batches:
            break
        data = data.to(device)
        targets = targets.to(device)
        spk_rec = forward_pass(net, data, flatten_input=flatten_input)
        acc = SF.accuracy_rate(spk_rec, targets)
        total += float(acc)
        count += 1

    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="snnTorch Tutorial 7 NMNIST example")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache/nmnist"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument(
        "--num-iters",
        type=int,
        default=50,
        help="Max batches per epoch. 0 walks the whole training set.",
    )
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--time-window-us", type=int, default=1000)
    parser.add_argument("--denoise-filter-time-us", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=Path, default=HERE / "weights.pt")
    parser.add_argument("--weights", type=Path, default=None, help="Optional checkpoint to load before training")
    parser.add_argument("--output", type=Path, default=None, help="Alias for --save")
    parser.add_argument(
        "--arch",
        type=str,
        default="mlir",
        choices=["csnn", "mlir"],
        help="csnn=tutorial conv/pool net, mlir=Linear/LIF exportable net",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    args = parser.parse_args()

    if args.output is not None:
        args.save = args.output

    torch.manual_seed(args.seed)
    device = _resolve_device()
    print(f"Using device: {device}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    trainloader, testloader = build_dataloaders(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        time_window_us=args.time_window_us,
        denoise_filter_time_us=args.denoise_filter_time_us,
    )

    flatten_input = args.arch == "mlir"
    if args.arch == "csnn":
        net = build_csnn_network(beta=args.beta).to(device)
    else:
        input_size = int(np.prod(tonic.datasets.NMNIST.sensor_size))
        net = build_mlir_network(input_size=input_size, hidden_size=args.hidden_size, beta=args.beta).to(device)

    if args.weights is not None and args.weights.exists():
        state = torch.load(args.weights, map_location="cpu")
        net.load_state_dict(state, strict=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)  # type: ignore[arg-type]

    loss_hist: list[float] = []
    acc_hist: list[float] = []

    for epoch in range(args.epochs):
        net.train()
        for i, (data, targets) in enumerate(trainloader):
            if args.num_iters > 0 and i >= args.num_iters:
                break

            data = data.to(device)
            targets = targets.to(device)

            spk_rec = forward_pass(net, data, flatten_input=flatten_input)
            loss_val = loss_fn(spk_rec, targets)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            acc = float(SF.accuracy_rate(spk_rec, targets))
            loss_hist.append(float(loss_val.item()))
            acc_hist.append(acc)

            print(
                f"Epoch {epoch} Iteration {i} "
                f"Train Loss: {loss_val.item():.4f} "
                f"Accuracy: {acc * 100.0:.2f}%"
            )

        test_acc = evaluate(
            net,
            testloader,
            device=device,
            max_batches=args.eval_batches,
            flatten_input=flatten_input,
        )
        print(f"Epoch {epoch} Eval Accuracy ({args.eval_batches} batches): {test_acc * 100.0:.2f}%")

    if args.save is not None and (args.epochs > 0 or args.num_iters > 0):
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), args.save)
        print(f"Saved model weights to: {args.save}")

    if loss_hist:
        print(f"Final train loss: {loss_hist[-1]:.4f}")
    if acc_hist:
        print(f"Final train accuracy: {acc_hist[-1] * 100.0:.2f}%")


if __name__ == "__main__":
    main()
