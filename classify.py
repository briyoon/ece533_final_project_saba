import argparse
import pathlib
import csv
import sys

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models
from torchvision.datasets import OxfordIIITPet


IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def load_classes():
    ds = OxfordIIITPet(
        root="data", split="test", target_types="category", download=True
    )
    return ds.classes


@torch.no_grad()
def predict(model, x, k=5):
    probs = model(x).softmax(dim=1)
    return probs.topk(k, dim=1)


def build_tfms(side):
    return T.Compose(
        [
            T.Resize(side, interpolation=Image.BICUBIC),
            T.CenterCrop(side),
            T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD),
        ]
    )


def load_model(ckpt_path: str, device: torch.device):
    model = models.resnet50(weights=None, num_classes=37)
    model.fc = torch.nn.Linear(2048, 37)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def main():
    ap = argparse.ArgumentParser(
        description="Batch classify images with multiple models and save results to CSV"
    )
    ap.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="List of model checkpoints (.pth/.pth.tar)",
    )
    ap.add_argument(
        "--images", nargs="+", required=True, help="List of image files to classify"
    )
    ap.add_argument(
        "--size", type=int, default=384, help="Resize & crop side length (default: 384)"
    )
    ap.add_argument("--out", default="results.csv", help="Output CSV filepath")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes()
    tfms = build_tfms(args.size)

    with open(args.out, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        header = (
            ["model", "image"]
            + [f"class{i}" for i in range(1, 6)]
            + [f"prob{i}" for i in range(1, 6)]
        )
        writer.writerow(header)

        for ckpt in args.ckpts:
            model = load_model(ckpt, device)
            for img_path in args.images:
                path = pathlib.Path(img_path)
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:
                    print(f"[!] Could not open {path}: {e}", file=sys.stderr)
                    continue

                x = tfms(img).unsqueeze(0).to(device)
                vals, idxs = predict(model, x, k=5)

                row = [ckpt, path.name]
                for prob, idx in zip(vals[0], idxs[0]):
                    row.append(classes[idx])
                    row.append(f"{prob.item()*100:.2f}")
                writer.writerow(row)

    print(f"Done. Results written to {args.out}")


if __name__ == "__main__":
    main()
