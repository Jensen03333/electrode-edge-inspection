"""Train a compact skip-connected crack segmenter and export fixed-shape FP32 ONNX."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SIZE = (320, 240)  # Pillow orders width before height.


def conv_pair(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.BatchNorm2d(outputs), nn.ReLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.BatchNorm2d(outputs), nn.ReLU(),
    )


class CrackUNet(nn.Module):
    """Three resolutions, with shallow detail features returned to each decoder."""

    def __init__(self):
        super().__init__()
        self.enc1 = conv_pair(1, 8)
        self.enc2 = conv_pair(8, 16)
        self.bottleneck = conv_pair(16, 32)
        self.dec2 = conv_pair(32 + 16, 16)
        self.dec1 = conv_pair(16 + 8, 8)
        self.output = nn.Conv2d(8, 1, 1)

    def forward(self, image):
        first = self.enc1(image)
        second = self.enc2(F.max_pool2d(first, 2))
        low = self.bottleneck(F.max_pool2d(second, 2))
        up = F.interpolate(low, scale_factor=2, mode="nearest")
        middle = self.dec2(torch.cat((up, second), dim=1))
        up = F.interpolate(middle, scale_factor=2, mode="nearest")
        return self.output(self.dec1(torch.cat((up, first), dim=1)))


class CrackFrames(Dataset):
    """Cache resized byte arrays so repeated epochs do not decode from disk."""

    def __init__(self, manifest, root):
        self.samples = []
        self.groups = set()
        self.files = set()
        root = root.resolve()
        positive = 0
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"file", "mask", "group"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing file/mask/group columns in {manifest}")
            for row in reader:
                group = row["group"].strip()
                paths = [(root / row[key]).resolve() for key in ("file", "mask")]
                if not group or any(not path.is_relative_to(root) for path in paths):
                    raise ValueError(f"Invalid source group or path in {row}")
                if paths[0] in self.files:
                    raise ValueError(f"Duplicate image in {manifest}: {paths[0]}")
                self.files.add(paths[0])
                self.groups.add(group)
                with Image.open(paths[0]) as image, Image.open(paths[1]) as label:
                    if image.size != label.size:
                        raise ValueError(f"Image and mask sizes differ: {paths[0]}")
                    if label.mode not in ("1", "L") or not np.isin(np.asarray(label), [0, 1, 255]).all():
                        raise ValueError(f"Expected a binary crack mask: {paths[1]}")
                    pixels = np.array(image.convert("L").resize(SIZE, Image.Resampling.BILINEAR))
                    mask = np.array(label.resize(SIZE, Image.Resampling.NEAREST)) != 0
                self.samples.append((torch.from_numpy(pixels)[None], torch.from_numpy(mask)[None]))
                positive += int(mask.sum())
        if not self.samples:
            raise ValueError(f"No images found in {manifest}")
        self.positive_fraction = positive / (len(self.samples) * SIZE[0] * SIZE[1])

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


@torch.inference_mode()
def validation_dice(model, validation, device):
    model.eval()
    tp = fp = fn = 0
    for image, target in validation:
        predicted = model(image.to(device).float() / 255) >= 0
        target = target.to(device)
        tp += (predicted & target).sum().item()
        fp += (predicted & ~target).sum().item()
        fn += (~predicted & target).sum().item()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def export(checkpoint, output):
    model = CrackUNet().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    example = torch.zeros(1, 1, SIZE[1], SIZE[0])
    torch.onnx.export(
        model, example, str(output), input_names=["image"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"FP32 logits model exported to {output}", flush=True)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    training = CrackFrames(args.train, args.data_root)
    validation = CrackFrames(args.val, args.data_root)
    if training.groups & validation.groups or training.files & validation.files:
        raise ValueError("Training and validation overlap; keep every source group together")
    if not 0 < training.positive_fraction < 1 or not 0 < validation.positive_fraction < 1:
        raise ValueError("Both splits need annotated crack and background pixels")
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    model = CrackUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    fraction = training.positive_fraction
    weight = min(12.0, max(1.0, math.sqrt((1 - fraction) / fraction)))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    report = {
        "architecture": "CrackUNet channels 8-16-32", "parameters": sum(p.numel() for p in model.parameters()),
        "seed": args.seed, "torch_version": str(torch.__version__), "device": str(device),
        "training_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "train_frames": len(training), "val_frames": len(validation),
        "train_groups": len(training.groups), "val_groups": len(validation.groups),
        "train_manifest_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "val_manifest_sha256": hashlib.sha256(args.val.read_bytes()).hexdigest(),
        "input_hw": [SIZE[1], SIZE[0]], "train_foreground_fraction": fraction,
        "bce_positive_weight": weight, "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "validation_metric": "Micro pixel Dice at training resolution, logits >= 0",
        "history": [],
    }
    best = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for image, target in loader:
            image = image.to(device).float() / 255
            target = target.to(device).float()
            if random.random() < 0.5:
                image, target = image.flip(-1), target.flip(-1)
            logits = model(image)
            prob = torch.sigmoid(logits)
            dice_loss = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
            loss = bce(logits, target) + dice_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * image.shape[0]
        dice = validation_dice(model, val_loader, device)
        epoch_loss = total_loss / len(training)
        report["history"].append({"epoch": epoch, "loss": epoch_loss, "val_dice_micro": dice})
        if dice > best:
            best = dice
            torch.save(model.state_dict(), args.output / "best.pt")
            report.update(best_epoch=epoch, best_val_dice_micro=dice)
        (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Epoch {epoch}/{args.epochs}: loss={epoch_loss:.4f} val_dice={dice:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("train", "val", "data-root"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-only", action="store_true", help="Export an existing output/best.pt")
    mode.add_argument("--no-export", action="store_true", help="Train now and export later")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or not (0 < args.learning_rate < math.inf):
        parser.error("epochs, batch size, threads and learning rate must be positive")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.export_only:
        if not all((args.train, args.val, args.data_root)):
            parser.error("Training requires --train, --val and --data-root")
        train(args)
    if not args.no_export:
        export(args.output / "best.pt", args.output / "fp32.onnx")


if __name__ == "__main__":
    main()
