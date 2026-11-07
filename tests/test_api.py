import io

from PIL import Image

KEY = "test-key"


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 200, 80)).save(buf, "PNG")
    return buf.getvalue()


def _upload(client, data=None, name="t.png", key=KEY):
    data = _png_bytes() if data is None else data
    headers = {"X-API-Key": key} if key else {}
    return client.post("/api/predict", data={"image": (io.BytesIO(data), name)},
                       headers=headers, content_type="multipart/form-data")


def test_api_disabled_when_no_key_is_configured(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    r = _upload(client)
    assert r.status_code == 503
    assert r.get_json()["error"]["code"] == "api_disabled"


def test_api_rejects_missing_and_wrong_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    assert _upload(client, key=None).status_code == 401
    assert _upload(client, key="wrong").status_code == 401


def test_api_classifies_a_valid_image(client, monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    r = _upload(client)
    assert r.status_code == 200
    body = r.get_json()
    assert body["label"] in {"AI", "Real"}
    assert 0.0 <= body["p_ai"] <= 1.0
    assert body["model_name"] and body["request_id"]


def test_api_rejects_wrong_type_fake_image_and_missing_file(client, monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    assert _upload(client, data=b"hello", name="notes.txt").status_code == 415
    assert _upload(client, data=b"not really a png", name="fake.png").status_code == 422
    r = client.post("/api/predict", data={}, headers={"X-API-Key": KEY},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_api_rejects_oversize_file_with_json(client, monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    r = _upload(client, data=b"0" * (6 * 1024 * 1024), name="big.jpg")
    assert r.status_code == 413
    assert r.get_json()["error"]["code"] == "file_too_large"


def test_health_stays_open_without_a_key(client):
    assert client.get("/health").status_code == 200