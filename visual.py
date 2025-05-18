import os
import re
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

from tqdm import tqdm
from PIL import Image

from datasets import BaselinePet, MaskedPet, FullAugPet, BackgroundAugPet
import albumentations as A
from albumentations.pytorch import ToTensorV2

from serverity import SEVERITIES

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def save_grid(dataset, path, n=9, size=3, unnorm=False, rand=True):
    if rand:
        idxs = torch.randperm(len(dataset))[:n]
    else:
        idxs = range(n)
    mean = torch.tensor(IMNET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMNET_STD).view(3, 1, 1)

    imgs = []
    for i in idxs:
        img = dataset[i][0]
        imgs.append(img if unnorm else (img * std + mean).clamp(0, 1))
    grid = vutils.make_grid(imgs, nrow=size, padding=2)
    vutils.save_image(grid, path)


def plot_metrics(run_dirs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for mode, run_dir in run_dirs.items():
        csv_path = os.path.join(run_dir, "log.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: no log.csv for {mode} at {run_dir}")
            continue
        df = pd.read_csv(csv_path)
        epochs = df["epoch"]

        fig, ax1 = plt.subplots()
        ax1.plot(epochs, df["loss"], marker="o", label="Train Loss")
        ax1.plot(epochs, df["val_loss"], marker="o", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax2 = ax1.twinx()
        ax2.plot(epochs, df["acc"], marker="x", ls="--", label="Train Acc")
        ax2.plot(epochs, df["val_acc"], marker="x", ls="--", label="Val Acc")
        ax2.set_ylabel("Accuracy")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")

        plt.title(f"{mode.capitalize()} Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{mode}_metrics.png"))
        plt.close(fig)


def sample_grids(data_root, size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ds_baseline = BaselinePet(data_root, "trainval", size, download=True)
    ds_masked = MaskedPet(data_root, "trainval", size, download=True)
    ds_full = FullAugPet(data_root, "trainval", size, download=True, p_aug=1.0)
    ds_bg = BackgroundAugPet(data_root, "trainval", size, download=True, p_aug=1.0)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    save_grid(ds_baseline, os.path.join(out_dir, "baseline_samples.png"), n=10, size=5)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    save_grid(ds_masked, os.path.join(out_dir, "masked_samples.png"), n=10, size=5)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    save_grid(ds_full, os.path.join(out_dir, "full_aug_samples.png"), n=10, size=5)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    save_grid(ds_bg, os.path.join(out_dir, "background_aug_samples.png"), n=10, size=5)


def build_corruption_transforms(size):
    transforms = {}
    baseline = A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
            ToTensorV2(),
        ]
    )
    for name, sev_list in SEVERITIES.items():
        tfms = [baseline]
        for lo, hi in sev_list:
            if name == "GaussNoise":
                op = A.GaussNoise(std_range=(lo, hi), p=1)
            elif name == "ShotNoise":
                op = A.ShotNoise(scale_range=(lo, hi), p=1)
            elif name == "ISONoise":
                op = A.ISONoise(intensity=(lo, hi), p=1)
            elif name == "GaussianBlur":
                op = A.GaussianBlur(sigma_limit=(lo, hi), p=1)
            elif name == "MotionBlur":
                op = A.MotionBlur(blur_limit=(int(lo), int(hi)), p=1)
            elif name == "Defocus":
                op = A.Defocus(radius=(int(lo), int(hi)), p=1)
            elif name == "ImageCompression":
                op = A.ImageCompression(
                    compression_type="jpeg", quality_range=(int(lo), int(hi)), p=1
                )
            elif name == "Brightness":
                op = A.RandomBrightnessContrast(
                    brightness_limit=(lo, hi), contrast_limit=0, p=1
                )
            elif name == "Contrast":
                op = A.RandomBrightnessContrast(
                    brightness_limit=0, contrast_limit=(lo, hi), p=1
                )
            elif name == "Pixelate":
                op = A.Downscale(scale_range=(lo, hi), p=1)
            else:
                raise ValueError(f"Unknown corruption {name}")
            tfms.append(
                A.Compose(
                    [
                        A.Resize(size, size),
                        op,
                        A.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
                        ToTensorV2(),
                    ]
                )
            )
        transforms[name] = tfms
    return transforms


def showcase_corruption_matrix(sample_img_path, size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_np = np.asarray(Image.open(sample_img_path).convert("RGB"))
    transforms = build_corruption_transforms(size)
    mean = torch.tensor(IMNET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMNET_STD).view(3, 1, 1)

    mats = []
    for sev in (1, 2, 3):
        for tfms in transforms.values():
            out = tfms[sev](image=img_np)["image"]
            t = out if isinstance(out, torch.Tensor) else torch.from_numpy(out)
            if t.ndim == 3 and t.shape[0] not in (1, 3):
                t = t.permute(2, 0, 1)
            mats.append((t * std + mean).clamp(0, 1))

    C = len(transforms)
    grid = vutils.make_grid(mats, nrow=C, padding=2)
    vutils.save_image(grid, os.path.join(out_dir, "corruption_matrix.png"))


def plot_eval_bars(eval_csv, out_dir):
    df = pd.read_csv(eval_csv)
    modes = ["baseline", "masked", "full_aug", "background_aug"]

    rows = {}
    for mode in modes:
        sel = df[df["ckpt"].str.contains(mode, case=False)]
        if len(sel) == 1:
            rows[mode] = sel.iloc[0]
        else:
            raise ValueError(
                f"Expected exactly one row for mode '{mode}', found {len(sel)}"
            )

    prefix = os.path.splitext(os.path.basename(eval_csv))[0]

    clean_vals = [rows[m]["clean"] for m in modes]
    fig, ax = plt.subplots()
    ax.bar(modes, clean_vals, color="C0")
    ax.set_ylabel("Clean Acc")
    ax.set_title("Clean Accuracy")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_clean_acc.png"))
    plt.close(fig)

    mR_vals = [rows[m]["mR"] for m in modes]
    fig, ax = plt.subplots()
    ax.bar(modes, mR_vals, color="C1")
    ax.set_ylabel("mR")
    ax.set_title("Mean Relative Robustness (mR)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_mR.png"))
    plt.close(fig)

    delta_vals = [rows[m]["delta_pp"] for m in modes]
    fig, ax = plt.subplots()
    ax.bar(modes, delta_vals, color="C2")
    ax.set_ylabel("ΔA (pp)")
    ax.set_title("Accuracy Gap (ΔA)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_delta_pp.png"))
    plt.close(fig)

    corr_cols = [c for c in df.columns if re.match(r".*-\d$", c)]
    breakdown = pd.DataFrame(
        {m: [rows[m][c] for c in corr_cols] for m in modes}, index=corr_cols
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    breakdown.plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Corruption Accuracy by Mode")
    ax.legend(title="Mode")
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_corruption_breakdown.png"))
    plt.close(fig)


def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument(
        "--runs",
        nargs=4,
        required=True,
        help="baseline, masked, full_aug, background_aug run dirs",
    )
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--out", default="visualizations")
    p.add_argument(
        "--sample_img",
        required=True,
        help="path to sample image for corruption showcase",
    )
    p.add_argument(
        "--eval_csv",
        nargs="+",
        default=[],
        help="one or more eval CSV files to visualize",
    )
    args = p.parse_args()

    modes = ["baseline", "masked", "full_aug", "background_aug"]
    run_dirs = dict(zip(modes, args.runs))

    plot_metrics(run_dirs, args.out)
    sample_grids(args.data, args.size, args.out)

    showcase_corruption_matrix(args.sample_img, args.size, args.out)

    for csv_path in args.eval_csv:
        plot_eval_bars(csv_path, args.out)


if __name__ == "__main__":
    main()
