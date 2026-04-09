from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from utils import list_images, load_mask_image, load_rgb_image, paired_random_flip, preprocess_image, preprocess_mask, stem_lookup


class AerialHouseDataset(Dataset):
    def __init__(self, split_dir: str | Path, image_size: int = 256, augment: bool = False) -> None:
        split_dir = Path(split_dir)
        image_paths = list_images(split_dir / "images")
        mask_paths = list_images(split_dir / "masks")
        mask_lookup = stem_lookup(mask_paths)
        self.samples = [(img, mask_lookup[img.stem]) for img in image_paths if img.stem in mask_lookup]
        self.image_size = image_size
        self.augment = augment

        if not self.samples:
            raise ValueError(f"No paired image/mask samples found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = load_rgb_image(image_path)
        mask = load_mask_image(mask_path)

        if self.augment:
            image, mask = paired_random_flip(image, mask)

        image_tensor = preprocess_image(image, self.image_size)
        mask_tensor = preprocess_mask(mask, self.image_size)
        return image_tensor, mask_tensor, image_path.name
