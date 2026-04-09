from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from utils import ensure_dir, list_images, save_json, set_seed, stem_lookup


def label_to_binary_mask(label_path: Path, house_class_id: int | None, house_color: tuple[int, int, int] | None, tolerance: int) -> np.ndarray:
    label_image = Image.open(label_path)
    label_np = np.asarray(label_image)

    if label_np.ndim == 2:
        if house_class_id is None:
            house_class_id = 1
        mask = (label_np == house_class_id).astype(np.uint8) * 255
        return mask

    if label_np.ndim == 3:
        if house_color is None:
            house_color = (255, 255, 255)
        house_rgb = np.asarray(house_color).reshape(1, 1, 3)
        diff = np.abs(label_np[..., :3].astype(np.int16) - house_rgb.astype(np.int16))
        mask = np.all(diff <= tolerance, axis=-1).astype(np.uint8) * 255
        return mask

    raise ValueError(f"Unsupported label shape for {label_path}: {label_np.shape}")


def copy_split(items: list[tuple[Path, Path]], split_name: str, output_dir: Path) -> None:
    image_out = ensure_dir(output_dir / split_name / "images")
    mask_out = ensure_dir(output_dir / split_name / "masks")

    for image_path, mask_path in items:
        shutil.copy2(image_path, image_out / image_path.name)
        shutil.copy2(mask_path, mask_out / mask_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate binary house masks and split the dataset.")
    parser.add_argument("--raw-images-dir", required=True, help="Folder containing aerial images.")
    parser.add_argument("--raw-labels-dir", required=True, help="Folder containing pixel labels or RGB masks.")
    parser.add_argument("--output-dir", default="data/processed", help="Folder to write train/val/test splits.")
    parser.add_argument("--mask-dir-name", default="generated_masks", help="Temporary folder name for generated masks.")
    parser.add_argument("--house-class-id", type=int, default=None, help="House class id when labels are grayscale class maps.")
    parser.add_argument("--house-color", nargs=3, type=int, default=None, metavar=("R", "G", "B"), help="House RGB label color for RGB label images.")
    parser.add_argument("--tolerance", type=int, default=0, help="Allowed color tolerance for RGB matching.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    set_seed(args.seed)

    raw_images = list_images(args.raw_images_dir)
    raw_labels = list_images(args.raw_labels_dir)
    label_lookup = stem_lookup(raw_labels)

    if not raw_images:
        raise ValueError("No raw images found.")

    output_dir = Path(args.output_dir)
    generated_masks_dir = ensure_dir(output_dir / args.mask_dir_name)

    paired_items: list[tuple[Path, Path]] = []
    for image_path in raw_images:
        label_path = label_lookup.get(image_path.stem)
        if not label_path:
            continue
        mask_np = label_to_binary_mask(
            label_path=label_path,
            house_class_id=args.house_class_id,
            house_color=tuple(args.house_color) if args.house_color else None,
            tolerance=args.tolerance,
        )
        mask_path = generated_masks_dir / f"{image_path.stem}.png"
        Image.fromarray(mask_np, mode="L").save(mask_path)
        paired_items.append((image_path, mask_path))

    if not paired_items:
        raise ValueError("No paired images/labels were found. Check filenames and directories.")

    random.shuffle(paired_items)
    total = len(paired_items)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    train_items = paired_items[:train_end]
    val_items = paired_items[train_end:val_end]
    test_items = paired_items[val_end:]

    copy_split(train_items, "train", output_dir)
    copy_split(val_items, "val", output_dir)
    copy_split(test_items, "test", output_dir)

    stats = {
        "total_samples": total,
        "train": len(train_items),
        "val": len(val_items),
        "test": len(test_items),
        "house_class_id": args.house_class_id,
        "house_color": args.house_color,
        "tolerance": args.tolerance,
        "note": "Masks are binary: house=255, background=0.",
    }
    save_json(stats, output_dir / "dataset_stats.json")
    print(f"Dataset ready. Stats saved to {output_dir / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
