import argparse
import os
import sys
import random
import csv
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from datasets import BaselinePet, MaskedPet, FullAugPet, BackgroundAugPet
from visual import save_grid

torch.set_float32_matmul_precision("high")


def build_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 37)
    return m


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument(
        "--mode",
        choices=["baseline", "masked", "full_aug", "background_aug"],
        default="baseline",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--p_aug", type=float, default=0.5)
    ap.add_argument("--samples", type=bool, default=False)
    return ap.parse_args()


def main():
    args = parse()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.mode == "baseline":
        train_ds = BaselinePet(args.data, "trainval", args.size, download=True)
    elif args.mode == "masked":
        train_ds = MaskedPet(args.data, "trainval", args.size, download=True)
    elif args.mode == "full_aug":
        train_ds = FullAugPet(
            args.data, "trainval", args.size, download=True, p_aug=args.p_aug
        )
    elif args.mode == "background_aug":
        train_ds = BackgroundAugPet(
            args.data, "trainval", args.size, download=True, p_aug=args.p_aug
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    val_ds = BaselinePet(args.data, "test", args.size, download=True)

    run = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + args.mode
    os.makedirs(f"runs/{run}", exist_ok=True)

    with open(os.path.join(f"runs/{run}", "params.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    save_grid(train_ds, f"runs/{run}/train_samples.png")

    if args.samples:
        sys.exit(0)

    train_loader = DataLoader(
        train_ds, args.batch, True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_ds, args.batch, False, num_workers=4, pin_memory=True)

    model = build_model().to(device)
    model = torch.compile(model, backend="inductor", fullgraph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criteron = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    csv_writer = csv.writer(open(f"runs/{run}/log.csv", "w", newline=""))
    csv_writer.writerow(["epoch", "loss", "acc", "val_loss", "val_acc"])

    for epoch in range(args.epochs):
        model.train()
        seen = correct = loss_sum = 0
        bar = tqdm(
            train_loader,
            ncols=120,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            total=len(train_loader),
        )
        for i, (inputs, labels) in enumerate(bar):
            inputs, labels = inputs.to(device), labels.to(device)
            prediction = model(inputs)
            loss = criteron(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            b = labels.size(0)
            seen += b
            loss_sum += loss.item() * b
            correct += (prediction.argmax(1) == labels).sum().item()
            bar.set_postfix(loss=f"{loss_sum/seen:.3f}", acc=f"{correct/seen*100:.2f}%")

        scheduler.step()

        model.eval()
        val_seen = 0
        val_correct = 0
        val_loss_sum = 0
        with torch.no_grad():
            bar = tqdm(
                val_loader,
                ncols=120,
                desc=f"Validation",
                total=len(val_loader),
            )
            for inputs, labels in bar:
                inputs, labels = inputs.to(device), labels.to(device)
                prediction = model(inputs)
                loss = criteron(prediction, labels)

                b = labels.size(0)
                val_seen += b
                val_loss_sum += loss.item() * b
                val_correct += (prediction.argmax(1) == labels).sum().item()
                bar.set_postfix(
                    val_loss=f"{val_loss_sum/seen:.3f}",
                    val_acc=f"{val_correct/seen*100:.2f}%",
                )

        train_loss = loss_sum / seen
        train_accuracy = correct / seen
        val_loss = val_loss_sum / val_seen
        val_accuracy = val_correct / val_seen
        csv_writer.writerow(
            [
                epoch + 1,
                f"{train_loss:.4f}",
                f"{train_accuracy:.4f}",
                f"{val_loss:.4f}",
                f"{val_accuracy:.4f}",
            ]
        )
        state_dict = (
            model._orig_mod.state_dict()
            if hasattr(model, "_orig_mod")
            else model.state_dict()
        )
        torch.save(state_dict, f"runs/{run}/e{epoch+1:02d}.pth")


if __name__ == "__main__":
    main()
