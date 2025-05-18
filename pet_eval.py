from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, csv, os, albumentations as A, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm

from serverity import SEVERITIES
from visual import save_grid

torch.set_float32_matmul_precision("high")

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def build_augmentations(size):
    C = {}
    C["Clean"] = A.Compose(
        [
            A.Resize(height=size, width=size),
        ]
    )
    for sigma in range(3):
        C[f"GaussNoise-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.GaussNoise(
                    std_range=SEVERITIES["GaussNoise"][sigma],
                    p=1,
                ),
            ]
        )

        C[f"ShotNoise-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.ShotNoise(
                    scale_range=SEVERITIES["ShotNoise"][sigma],
                    p=1,
                ),
            ]
        )

        C[f"ISONoise-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.ISONoise(
                    intensity=SEVERITIES["ISONoise"][sigma],
                    p=1,
                ),
            ]
        )

        C[f"GaussBlur-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.GaussianBlur(
                    sigma_limit=SEVERITIES["GaussianBlur"][sigma],
                    p=1,
                ),
            ]
        )

        C[f"MotionBlur-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.MotionBlur(
                    blur_limit=SEVERITIES["MotionBlur"][sigma],
                    p=1,
                ),
            ]
        )

        C[f"Defocus-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.Defocus(
                    radius=SEVERITIES["Defocus"][sigma],
                    p=1,
                ),
            ]
        )

        C[f"ImageCompression-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.ImageCompression(
                    quality_range=SEVERITIES["ImageCompression"][sigma],
                    p=1,
                ),
            ]
        )

        C[f"Brightness-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.RandomBrightnessContrast(
                    brightness_limit=SEVERITIES["Brightness"][sigma],
                    contrast_limit=(0, 0),
                    p=1,
                ),
            ]
        )

        C[f"Contrast-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.RandomBrightnessContrast(
                    brightness_limit=(0, 0),
                    contrast_limit=SEVERITIES["Contrast"][sigma],
                    p=1,
                ),
            ]
        )

        C[f"Pixelate-{sigma+1}"] = A.Compose(
            [
                A.Resize(height=size, width=size),
                A.Downscale(
                    scale_range=SEVERITIES["Pixelate"][sigma],
                    p=1,
                ),
            ]
        )
    return C


def evaluate(model: models.ResNet, loader: DataLoader, device: torch.device, desc: str):
    with torch.no_grad():
        model.to(device)
        model.eval()

        correct = 0
        total = 0
        bar = tqdm(loader, desc=desc, ncols=120, total=len(loader))
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
            bar.set_postfix(acc=f"{correct/total*100:.2f}%")

    return correct / total


def build_cache(name, aug, root):
    path = os.path.join(root, f"{name}.npz")
    if os.path.exists(path):
        print("Cache already exists, skipping...")
        return

    print(f"Creating cache for {name}...")

    clean_ds = OxfordIIITPet(
        "data", split="test", target_types="category", download=True
    )

    xs, ys = [], []
    for img, lab in clean_ds:
        out = aug(image=np.asarray(img))["image"]
        out = np.transpose(out, (2, 0, 1))
        xs.append(out)
        ys.append(lab)

    xs = np.stack(xs, axis=0).astype(np.uint8)
    ys = np.asarray(ys, dtype=np.int16)

    print(f"Saving to {path}")
    np.savez(path, x=xs, y=ys)
    print("Done.")

    save_grid(
        TensorDataset(torch.from_numpy(xs).float().div_(255), torch.zeros(len(xs))),
        os.path.join(root, f"{name}_samples.png"),
        n=9,
        size=3,
        unnorm=True,
    )


def get_loader(name: str, cache_dir, batch):
    path = os.path.join(cache_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache {name} not found in {cache_dir}")

    print(f"Loading cached data {name}...")
    data = np.load(path, mmap_mode="r")
    xs = torch.from_numpy(data["x"]).float().div_(255.0)
    mean = torch.tensor(IMNET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMNET_STD).view(3, 1, 1)
    xs = (xs - mean).div_(std)
    ys = torch.as_tensor(data["y"], dtype=torch.int16)

    return DataLoader(
        TensorDataset(xs, ys),
        batch_size=batch,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpts", nargs="+", required=True, help="list of model checkpoints"
    )
    ap.add_argument("--data", default="./data")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=str, required=True, help="output directory")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    cache_dir = os.path.join("data", "eval")
    os.makedirs(cache_dir, exist_ok=True)

    augmentations = build_augmentations(args.size)
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [
            ex.submit(build_cache, name, aug, cache_dir)
            for name, aug in augmentations.items()
        ]
        for fut in as_completed(futures):
            fut.result()

    log_path = args.out
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            header = ["ckpt", "clean"] + list(augmentations.keys()) + ["mR", "delta_pp"]
            writer.writerow(header)

        for ckpt in args.ckpts:
            print(f"{"_".join(os.path.relpath(ckpt).split("/")[-2:])}:")

            model = models.resnet50(weights=None, num_classes=37)
            model.fc = torch.nn.Linear(2048, 37)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model = torch.compile(model)

            accs = []
            for name, _ in augmentations.items():
                loader = get_loader(name, cache_dir, args.batch)
                acc = evaluate(model, loader, device, name)
                accs.append(acc)
                del loader
                torch.cuda.empty_cache()

            acc_clean = accs[0]
            acc_corr = accs[1:]
            mR = float(np.mean(np.array(acc_corr) / acc_clean))
            delta = acc_clean - float(np.mean(acc_corr))
            print(f"=> Clean {acc_clean*100:.2f}%  mR {mR:.3f}  ΔA {delta*100:.1f} pp")

            row = (
                [ckpt, f"{acc_clean:.4f}"]
                + [f"{a:.4f}" for a in accs]
                + [f"{mR:.4f}", f"{delta:.4f}"]
            )
            writer.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
