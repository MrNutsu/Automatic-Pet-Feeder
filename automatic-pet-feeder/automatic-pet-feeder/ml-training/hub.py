"""Feeder test hub: pick a model, then test with the webcam or analyze an image.

    python hub.py

Needs:  pip install ultralytics opencv-python
Linux:  sudo apt install python3-tk      (only if Python says tkinter is missing)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # models and downloads live next to this file
WANTED = {"cat", "cats", "dog", "person"}  # 'car' may exist in the model but is ignored on purpose
PRETRAINED = "yolo11n.pt"  # downloaded automatically the first time
IMG_TYPES = [("Images", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")]


def trained_weights():
    """Newest best.pt created by data_training.py --train, or None."""
    found = sorted(HERE.glob("runs/**/best.pt"), key=lambda f: f.stat().st_mtime)
    return str(found[-1]) if found else None


# ----------------------------------------------------------------- webcam (separate process)
def open_camera(preferred):
    import cv2

    for idx in dict.fromkeys([preferred, 0, 1, 2, 3]):  # first one that really delivers frames
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)
        if cap.isOpened() and cap.read()[0]:
            print(f"Using camera {idx}")
            return cap
        cap.release()
    sys.exit("No working camera found. Close other apps using it (browser, Zoom) and check permissions.")


def run_webcam(weights, conf, cam):
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    ids = [i for i, n in model.names.items() if n.lower() in WANTED]
    if not ids:
        sys.exit(f"This model has no cat/dog/person classes: {model.names}")
    cap = open_camera(cam)
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, conf=conf, classes=ids, verbose=False)[0]
        found = sorted({model.names[int(c)] for c in r.boxes.cls})
        if found != last:  # only print when something changes
            print("Detected:", found or "nothing")
            last = found
        try:
            cv2.imshow("Webcam test - press q to quit", r.plot())
        except cv2.error:
            sys.exit("OpenCV has no window support. Run: pip uninstall -y opencv-python-headless && pip install opencv-python")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------- the hub window
def run_hub():
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError:
        sys.exit("Tkinter is missing. Install it with: sudo apt install python3-tk")
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Automatic feeder - test hub")
    root.geometry("820x740")
    ttk.Style().configure("Big.TButton", font=("Helvetica", 13), padding=10)

    trained = trained_weights()
    choice = tk.StringVar(value="trained" if trained else "pretrained")
    conf = tk.DoubleVar(value=0.5)
    conf_text = tk.StringVar(value="0.50")
    conf.trace_add("write", lambda *_: conf_text.set(f"{conf.get():.2f}"))
    status = tk.StringVar(value="Ready.")
    models, state = {}, {"proc": None, "photo": None}

    def weights():
        return trained if choice.get() == "trained" else PRETRAINED

    def get_model(path):
        if path not in models:
            status.set("Loading model (the first time takes a few seconds)...")
            root.update_idletasks()
            from ultralytics import YOLO

            m = YOLO(path)
            models[path] = (m, [i for i, n in m.names.items() if n.lower() in WANTED])
        return models[path]

    def start_webcam():
        p = state["proc"]
        if p and p.poll() is None:
            status.set("The webcam is already running - press q in its window to close it.")
            return
        state["proc"] = subprocess.Popen([
            sys.executable, str(Path(__file__).resolve()), "--webcam",
            "--weights", weights(), "--conf", f"{conf.get():.2f}",
        ])
        status.set("Webcam window opening... press q in it to close it. (Confidence is read when it starts.)")

    def analyze_image():
        file = filedialog.askopenfilename(title="Choose an image", filetypes=IMG_TYPES)
        if not file:
            return
        try:
            model, ids = get_model(weights())
            status.set("Analyzing...")
            root.update_idletasks()
            r = model(file, conf=conf.get(), classes=ids, verbose=False)[0]
            found = [f"{model.names[int(b.cls)]} {float(b.conf):.2f}" for b in r.boxes]
            img = Image.fromarray(r.plot()[:, :, ::-1])  # BGR -> RGB
            img.thumbnail((780, 500))
            state["photo"] = ImageTk.PhotoImage(img)  # keep a reference or Tk drops the picture
            picture.config(image=state["photo"], text="")
            status.set("Detected: " + ", ".join(found) if found
                       else "Nothing detected. Try lowering the confidence slider.")
        except Exception as e:  # e.g. corrupted image
            status.set(f"Could not analyze that file: {e}")

    def on_close():
        p = state["proc"]
        if p and p.poll() is None:
            p.terminate()
        root.destroy()

    ttk.Label(root, text="Automatic feeder - test hub", font=("Helvetica", 18, "bold")).pack(pady=(14, 6))

    box = ttk.LabelFrame(root, text="Model")
    box.pack(padx=16, pady=6, fill="x")
    ttk.Radiobutton(box, text="Pretrained YOLO11 (knows cat, dog, person)",
                    variable=choice, value="pretrained").pack(anchor="w", padx=8, pady=2)
    ttk.Radiobutton(box, text="My trained model" + ("" if trained else "  (not found - run data_training.py --train first)"),
                    variable=choice, value="trained",
                    state="normal" if trained else "disabled").pack(anchor="w", padx=8, pady=2)

    row = ttk.Frame(root)
    row.pack(pady=6)
    ttk.Label(row, text="Minimum confidence").pack(side="left")
    ttk.Scale(row, from_=0.1, to=0.9, variable=conf, length=220).pack(side="left", padx=8)
    ttk.Label(row, textvariable=conf_text, width=5).pack(side="left")

    buttons = ttk.Frame(root)
    buttons.pack(pady=8)
    ttk.Button(buttons, text="Webcam test", style="Big.TButton", command=start_webcam).pack(side="left", padx=10)
    ttk.Button(buttons, text="Analyze an image", style="Big.TButton", command=analyze_image).pack(side="left", padx=10)

    picture = ttk.Label(root, text="Choose an image to see the result here", anchor="center")
    picture.pack(expand=True, fill="both", padx=16, pady=8)
    ttk.Label(root, textvariable=status, font=("Helvetica", 12), wraplength=780).pack(pady=(0, 14))

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    os.chdir(HERE)
    p = argparse.ArgumentParser()
    p.add_argument("--webcam", action="store_true", help="used internally by the hub")
    p.add_argument("--weights", default=PRETRAINED)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--cam", type=int, default=0)
    a = p.parse_args()
    if a.webcam:
        run_webcam(a.weights, a.conf, a.cam)
    else:
        run_hub()
