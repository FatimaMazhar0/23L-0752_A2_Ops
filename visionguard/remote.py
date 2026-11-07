import json
import urllib.error
import urllib.request
import uuid

from . import config


class RemoteError(Exception):
    def __init__(self, message, status=502, title="Model service unavailable"):
        super().__init__(message)
        self.status, self.title = status, title


def _multipart(filename, data):
    boundary = uuid.uuid4().hex
    head = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n").encode()
    return boundary, head + data + f"\r\n--{boundary}--\r\n".encode()


def _error_text(err):
    try:
        body = json.loads(err.read().decode("utf-8", "replace"))
        e = body.get("error")
        if isinstance(e, dict):
            return e.get("message") or e.get("code") or ""
        return body.get("message") or (e if isinstance(e, str) else "")
    except Exception:
        return ""


def predict(data, filename):
    """Send the image to the hosted API. Returns (result, meta)."""
    if not config.MODEL_API_URL:
        raise RemoteError("MODEL_API_URL is not set.", 500, "Configuration error")
    boundary, body = _multipart(filename, data)
    req = urllib.request.Request(
        f"{config.MODEL_API_URL}/api/predict", data=body, method="POST",
        headers={"X-API-Key": config.MODEL_API_KEY,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=config.MODEL_API_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RemoteError("The model service rejected our API key.", 502) from e
        if e.code in (400, 413, 415, 422):
            raise RemoteError(_error_text(e) or "The model service could not use that image.",
                              e.code, "The model service could not use that image") from e
        raise RemoteError(f"The model service returned an error ({e.code}).", 502) from e
    except TimeoutError as e:
        raise RemoteError("The model service took too long to answer. "
                          "It may be starting up; please try again.", 504) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise RemoteError("The model service took too long to answer.", 504) from e
        raise RemoteError("Could not reach the model service.", 502) from e
    try:
        j = json.loads(raw)
        result = {"label": j["label"], "p_ai": float(j["p_ai"]),
                  "confidence": float(j["confidence"]), "threshold": float(j["threshold"])}
        meta = {"model_name": j["model_name"], "model_version": str(j["model_version"])}
    except (KeyError, ValueError, TypeError) as e:
        raise RemoteError("The model service sent an unexpected response.", 502) from e
    return result, meta