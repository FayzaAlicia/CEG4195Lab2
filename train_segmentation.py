from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import AerialHouseDataset
from metrics import DiceBCELoss, dice_score, iou_score
from model import UNet
from utils import ensure_dir, save_json, set_seed


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    criterion = DiceBCELoss()
    total_iou = 0.0
    total_dice = 0.0
    batches = 0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        total_loss += float(loss.item())
        total_iou += iou_score(preds, masks)
        total_dice += dice_score(preds, masks)
        batches += 1

    return {
        "loss": total_loss / max(batches, 1),
        "iou": total_iou / max(batches, 1),
        "dice": total_dice / max(batches, 1),
    }


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["val_iou"], label="Val IoU")
    plt.plot(history["val_dice"], label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


@torch.no_grad()
def save_sample_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float, output_path: Path, max_samples: int = 3) -> None:
    model.eval()
    rows = []

    for images, masks, _ in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float().cpu()

        for i in range(min(images.size(0), max_samples - len(rows))):
            image = images[i].cpu().permute(1, 2, 0).numpy()
            mask = masks[i].cpu().squeeze(0).numpy()
            pred = preds[i].squeeze(0).numpy()
            rows.append((image, mask, pred))
        if len(rows) >= max_samples:
            break

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for row_idx, (image, mask, pred) in enumerate(rows):
        axes[row_idx][0].imshow(image)
        axes[row_idx][0].set_title("Image")
        axes[row_idx][1].imshow(mask, cmap="gray")
        axes[row_idx][1].set_title("Ground Truth")
        axes[row_idx][2].imshow(pred, cmap="gray")
        axes[row_idx][2].set_title("Prediction")
        for col in range(3):
            axes[row_idx][col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a U-Net for aerial house segmentation.")
    parser.add_argument("--data-dir", default="data/processed", help="Folder containing train/val/test splits.")
    parser.add_argument("--output-dir", default="artifacts", help="Folder to save checkpoints and figures.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = AerialHouseDataset(Path(args.data_dir) / "train", image_size=args.image_size, augment=True)
    val_ds = AerialHouseDataset(Path(args.data_dir) / "val", image_size=args.image_size, augment=False)
    test_ds = AerialHouseDataset(Path(args.data_dir) / "test", image_size=args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    best_val_iou = -1.0
    checkpoint_path = output_dir / "model_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        train_loss = running_loss / max(batches, 1)
        val_metrics = evaluate(model, val_loader, device, args.threshold)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f}"
        )

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, args.threshold)

    save_training_curves(history, output_dir / "training_curves.png")
    save_sample_predictions(model, test_loader, device, args.threshold, output_dir / "sample_predictions.png")

    final_metrics = {
        "best_val_iou": best_val_iou,
        "test_loss": test_metrics["loss"],
        "test_iou": test_metrics["iou"],
        "test_dice": test_metrics["dice"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.lr,
    }
    save_json(final_metrics, output_dir / "metrics.json")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
