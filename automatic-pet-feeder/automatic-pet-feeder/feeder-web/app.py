"""Feeder web test: one page where visitors upload a photo and see what the model finds.

Only two routes exist: GET / (the page) and POST /predict (the analysis).
Photos are processed in memory and never written to disk.
"""
import base64
import io
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # reject anything bigger than 10 MB
MAX_SIDE = 1280                      # photos are shrunk to this before analysis
CONF = 0.5                           # minimum confidence to report a detection
WANTED = {"cat", "cats", "dog", "person"}  # 'car' may exist in the model but is ignored

# Put your own trained file at models/best.pt to use it; otherwise the pretrained model is used.
WEIGHTS = "models/best.pt" if Path("models/best.pt").exists() else "yolo11n.pt"
model = YOLO(WEIGHTS)
IDS = [i for i, n in model.names.items() if n.lower() in WANTED]
print(f"Model loaded: {WEIGHTS} | classes shown: {[model.names[i] for i in IDS]}", flush=True)

lock = threading.Lock()  # the model handles one photo at a time; other visitors wait their turn
app = Flask(__name__, static_folder=None)  # no /static route: nothing else is exposed
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No photo received."), 400
    try:
        img = ImageOps.exif_transpose(Image.open(f.stream)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not a valid image."), 400

    img.thumbnail((MAX_SIDE, MAX_SIDE))
    with lock:
        r = model(img, conf=CONF, classes=IDS, verbose=False)[0]
        plotted = r.plot()  # BGR array with boxes drawn
    detections = [
        {"label": model.names[int(b.cls)].replace("cats", "cat"), "confidence": round(float(b.conf), 2)}
        for b in r.boxes
    ]

    buf = io.BytesIO()
    Image.fromarray(plotted[:, :, ::-1]).save(buf, "JPEG", quality=85)
    resp = jsonify(
        detections=detections,
        image="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That photo is too big (10 MB maximum)."), 413


@app.errorhandler(Exception)
def unexpected(e):
    code = getattr(e, "code", 500)
    if code == 404 or code == 405:
        return jsonify(error="Not found."), code
    print("Error:", repr(e), flush=True)
    return jsonify(error="Something went wrong. Please try another photo."), 500
