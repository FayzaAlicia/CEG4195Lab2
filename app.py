from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from PIL import Image

from model import UNet
from utils import getenv_bool, mask_to_base64_png, maybe_download_file, preprocess_image


def load_model() -> tuple[UNet, str, torch.device]:
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/model_best.pth"))
    model_url = os.getenv("MODEL_URL", "")
    hf_token = os.getenv("HF_TOKEN", "")
    strict_model_load = getenv_bool("STRICT_MODEL_LOAD", default=False)

    device = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    model = UNet().to(device)
    model_status = "random_weights"

    if not model_path.exists() and model_url:
        maybe_download_file(model_url, model_path, token=hf_token or None)

    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model_status = "trained_checkpoint"
    elif strict_model_load:
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path}. Train the model first or set MODEL_URL/HF_TOKEN in .env"
        )

    model.eval()
    return model, model_status, device


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__)
    model, model_status, device = load_model()
    image_size = int(os.getenv("IMAGE_SIZE", "256"))
    threshold = float(os.getenv("MASK_THRESHOLD", "0.5"))

    @app.get("/health")
    def health():
        return {"status": "ok", "model_status": model_status, "device": str(device)}

    @app.post("/predict")
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "Please upload an image file with form field name 'image'."}), 400

        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "Uploaded file is empty."}), 400

        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception as exc:
            return jsonify({"error": f"Invalid image file: {exc}"}), 400

        original_width, original_height = image.size
        image_tensor = preprocess_image(image, image_size).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask_small = (probs >= threshold).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_small, mode="L").resize((original_width, original_height), Image.NEAREST)
        mask_array = np.asarray(mask_image)
        house_pixels = int((mask_array > 0).sum())
        coverage = house_pixels / float(mask_array.size)

        return jsonify(
            {
                "filename": file.filename,
                "model_status": model_status,
                "mask_shape": list(mask_array.shape),
                "house_pixels": house_pixels,
                "coverage": round(coverage, 6),
                "mask_png_base64": mask_to_base64_png(mask_array),
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "5000"))
    app.run(host=host, port=port)
