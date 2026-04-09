from __future__ import annotations

import base64
import json
import os
import random
import shutil
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def list_images(folder: str | Path) -> list[Path]:
    folder_path = Path(folder)
    return sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES])


def stem_lookup(paths: Iterable[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    return torch.from_numpy(image_np)


def preprocess_mask(mask: Image.Image, image_size: int) -> torch.Tensor:
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    mask_np = np.asarray(mask, dtype=np.float32)
    if mask_np.max() > 1:
        mask_np = mask_np / 255.0
    mask_np = (mask_np >= 0.5).astype(np.float32)
    return torch.from_numpy(mask_np).unsqueeze(0)


def paired_random_flip(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.2:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return image, mask


def mask_to_base64_png(mask_array: np.ndarray) -> str:
    mask_img = Image.fromarray(mask_array.astype(np.uint8), mode="L")
    buffer = BytesIO()
    mask_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_json(data: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def maybe_download_file(url: str | None, destination: str | Path, token: str | None = None) -> bool:
    if not url:
        return False

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(request) as response, open(destination_path, "wb") as f:
        shutil.copyfileobj(response, f)

    return True


def getenv_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
