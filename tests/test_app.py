import io

from PIL import Image


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 180, 90)).save(buf, "PNG")
    return buf.getvalue()


def test_health_reports_model_and_database(client):
    r = client.get("/health")
    body = r.get_json()
    assert r.status_code == 200
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database"] == "ok"


def test_home_page_loads(client):
    assert client.get("/").status_code == 200


def test_valid_image_is_classified_and_saved(client):
    r = client.post(
        "/predict",
        data={"image": (io.BytesIO(_png_bytes()), "test.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    history = client.get("/history")
    assert history.status_code == 200
    assert b"test.png" in history.data


def test_wrong_extension_is_rejected(client):
    r = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"hello"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 415


def test_fake_image_is_rejected(client):
    r = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"this is not an image"), "fake.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 422


def test_missing_file_is_rejected(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
