import base64
import hashlib
import io
import logging

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from . import config, db

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_PIXELS = 50_000_000
CAVEAT = ("VisionGuard is a small model trained on a few hundred images. On its held-out test set "
          "it was right about 80% of the time, so it makes mistakes, and its confidence is not a "
          "guarantee. It leans on visual style (polished or painterly images) and is not proof that "
          "an image was generated. Do not treat this result as evidence on its own.")


class UploadError(Exception):
    def __init__(self, message, status=400, title="We could not use that file"):
        super().__init__(message)
        self.message, self.status, self.title = message, status, title


def _validate(file):
    """Return (clean_name, bytes) or raise UploadError. Nothing is written to disk."""
    if file is None or not file.filename:
        raise UploadError("Please choose an image file first.")
    name = (secure_filename(file.filename) or "upload")[:100]
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXT:
        raise UploadError(f"Files of type '.{ext or 'unknown'}' are not accepted. "
                          "Please upload a PNG, JPG or WEBP image.", 415, "Unsupported file type")
    data = file.read()
    if not data:
        raise UploadError("That file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
            if fmt not in ALLOWED_FORMATS:
                raise UploadError(f"The file's contents are {fmt}, not PNG, JPG or WEBP, "
                                  f"although its name ends in .{ext}.", 415, "Unsupported file type")
            if w * h > MAX_PIXELS:
                raise UploadError(f"That image is too large ({w} x {h} pixels).", 413, "Image too large")
            im.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise UploadError("That file looks like an image but could not be read. "
                          "It may be corrupted or not a real image.", 422, "Image could not be decoded")
    return name, data


def _preview(data):
    """Small in-memory thumbnail embedded in the page, so the image never touches the disk."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            im.thumbnail((400, 400))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def create_app():
    from . import inference                      # imported here so `python -m visionguard.inference` stays clean

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    db.init(config.DATABASE_URL)
    inference.load_model()                       # loaded once, at start-up
    from .api import bp as api_bp
    app.register_blueprint(api_bp)
    app.logger.info("Model loaded from %s | threshold %.2f", config.MODEL_PATH, config.THRESHOLD)

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=f"{config.MAX_UPLOAD_MB:g}")

    @app.post("/predict")
    def predict():
        name, data = _validate(request.files.get("image"))
        try:
            r = inference.predict(data)
        except inference.ImageDecodeError:
            raise UploadError("That image could not be decoded for analysis.", 422,
                              "Image could not be decoded")
        saved = True
        try:
            db.add_prediction(filename=name, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data),
                              model_name=config.MODEL_NAME, model_version=config.MODEL_VERSION, **r)
        except Exception:
            app.logger.exception("Could not save the prediction to the database")
            saved = False
        return render_template("result.html", r=r, filename=name, preview=_preview(data),
                               caveat=CAVEAT, model_name=config.MODEL_NAME,
                               model_version=config.MODEL_VERSION, saved=saved)
        
    @app.get("/history")
    def history():
        rows, total = db.recent(100)
        return render_template("history.html", rows=rows, total=total)

    @app.get("/health")
    def health():
        status = {"status": "ok", "model_loaded": inference.model_loaded(), "database": "ok",
                  "model": config.MODEL_NAME, "model_version": config.MODEL_VERSION,
                  "threshold": config.THRESHOLD, "debug": app.debug}
        try:
            db.ping()
        except Exception:
            status["database"] = "unreachable"
        ok = status["model_loaded"] and status["database"] == "ok"
        if not ok:
            status["status"] = "degraded"
        return jsonify(status), (200 if ok else 503)

    def _error(code, title, message):
        return render_template("error.html", code=code, title=title, message=message), code

    @app.errorhandler(UploadError)
    def _upload_error(e):
        return _error(e.status, e.title, e.message)

    @app.errorhandler(413)
    def _too_big(e):
        return _error(413, "File too large",
                      f"That file is larger than the {config.MAX_UPLOAD_MB:g} MB limit. "
                      "Please choose a smaller image.")

    @app.errorhandler(404)
    def _not_found(e):
        return _error(404, "Page not found", "That page does not exist.")

    @app.errorhandler(405)
    def _bad_method(e):
        return _error(405, "Not allowed", "That address only accepts uploads from the form on the home page.")

    @app.errorhandler(500)
    def _server_error(e):
        return _error(500, "Something went wrong",
                      "The server hit an unexpected problem. Please try again in a moment.")

    return app