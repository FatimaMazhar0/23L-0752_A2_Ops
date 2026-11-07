import io

from PIL import Image

from visionguard import config, create_app, remote


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 80, 200)).save(buf, "PNG")
    return buf.getvalue()


def test_remote_backend_uses_api_and_saves_result(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_BACKEND", "remote")
    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{tmp_path / 'remote.db'}")
    monkeypatch.setattr(remote, "predict", lambda data, filename: (
        {"label": "AI", "p_ai": 0.9, "confidence": 0.9, "threshold": 0.46},
        {"model_name": "demo-model", "model_version": "7"}))
    client = create_app().test_client()
    r = client.post("/predict", data={"image": (io.BytesIO(_png()), "x.png")})
    assert r.status_code == 200
    assert b"Likely AI-generated" in r.data
    assert b"demo-model (remote API)" in client.get("/history").data


def test_remote_failure_shows_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_BACKEND", "remote")
    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{tmp_path / 'fail.db'}")

    def boom(data, filename):
        raise remote.RemoteError("The model service rejected our API key.", 502)

    monkeypatch.setattr(remote, "predict", boom)
    client = create_app().test_client()
    r = client.post("/predict", data={"image": (io.BytesIO(_png()), "x.png")})
    assert r.status_code == 502
    assert b"Model service unavailable" in r.data