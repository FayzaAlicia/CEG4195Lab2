import io
import os

from PIL import Image

os.environ["STRICT_MODEL_LOAD"] = "false"
os.environ["MODEL_PATH"] = "artifacts/does_not_exist_for_tests.pth"

from app import create_app  # noqa: E402



def make_test_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


def test_health_endpoint():
    app = create_app()
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"


def test_predict_endpoint_accepts_image_upload():
    app = create_app()
    client = app.test_client()
    response = client.post(
        "/predict",
        data={"image": (io.BytesIO(make_test_png_bytes()), "sample.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["filename"] == "sample.png"
    assert "mask_png_base64" in data
    assert "coverage" in data
