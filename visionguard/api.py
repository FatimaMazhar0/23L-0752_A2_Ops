import hashlib
import hmac
import os
import time
import uuid

from flask import Blueprint, jsonify, request

from . import UploadError, _validate, config, inference

bp = Blueprint("api", __name__, url_prefix="/api")


def _json_error(status, code, message):
    return jsonify({"error": {"code": code, "message": message}}), status


@bp.before_request
def _require_key():
    expected = os.getenv("API_KEY", "")
    if not expected:
        return _json_error(503, "api_disabled", "The API is not enabled on this server.")
    given = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(given.encode(), expected.encode()):
        return _json_error(401, "invalid_api_key", "Missing or invalid X-API-Key header.")


@bp.post("/predict")
def predict():
    t0 = time.perf_counter()
    try:
        name, data = _validate(request.files.get("image"))
        r = inference.predict(data)
    except UploadError as e:
        return _json_error(e.status, "invalid_image", e.message)
    except inference.ImageDecodeError:
        return _json_error(422, "undecodable_image", "That image could not be decoded for analysis.")
    return jsonify({
        "request_id": uuid.uuid4().hex,
        "filename": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "label": r["label"],
        "p_ai": round(r["p_ai"], 6),
        "confidence": round(r["confidence"], 6),
        "threshold": r["threshold"],
        "model_name": config.MODEL_NAME,
        "model_version": config.MODEL_VERSION,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    })


@bp.errorhandler(413)
def _too_large(e):
    return _json_error(413, "file_too_large", f"The file is larger than {config.MAX_UPLOAD_MB:g} MB.")