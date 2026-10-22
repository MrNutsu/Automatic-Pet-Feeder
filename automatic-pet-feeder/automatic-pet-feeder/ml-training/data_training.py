"""Webcam test for the cat / dog / person detector.

    python data_training.py                       webcam with the pretrained YOLO11 (no dataset needed)
    python data_training.py --train               unzip the dataset, train, then open the webcam with your new model
    python data_training.py --weights best.pt     webcam with a model you already trained

Install once:  pip install ultralytics opencv-python
Press q in the video window to quit.
"""
import argparse
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent  # everything is looked up next to this file
WANTED = {"cat", "cats", "dog", "person"}  # 'car' is in the dataset but ignored on purpose


def find_zip(zip_arg):
    if zip_arg:
        for c in (Path(zip_arg), HERE / zip_arg):
            if c.exists():
                return c
    else:
        found = sorted(HERE.glob("cat-dog-person*.zip"))
        if found:
            return found[0]
    files = ", ".join(p.name for p in HERE.iterdir())
    sys.exit(f"Could not find the dataset zip.\nLooked in: {HERE}\nFiles there: {files}\n"
             "Put the zip next to data_training.py or pass --zip <path>.")


def prepare_dataset(zip_arg):
    """Unzip once into ./dataset and write a data yaml with absolute paths."""
    import yaml

    out = HERE / "dataset"
    if not (out / "train" / "images").exists():
        z = find_zip(zip_arg)
        print(f"Unzipping {z.name} ...")
        with zipfile.ZipFile(z) as f:
            f.extractall(out)

    names = yaml.safe_load((out / "data.yaml").read_text())["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else dict(names)
    names = {int(i): ("cat" if n == "cats" else n) for i, n in names.items()}
    cfg = {"path": out.as_posix(), "train": "train/images", "val": "valid/images",
           "test": "test/images", "names": names}
    fixed = out / "data_fixed.yaml"
    fixed.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return fixed


def train(zip_arg, model_name, epochs):
    import torch
    from ultralytics import YOLO

    data = prepare_dataset(zip_arg)
    gpu = torch.cuda.is_available()
    print("Training on", "GPU" if gpu else "CPU - this can take a long time, consider Google Colab")
    model = YOLO(model_name)
    model.train(
        data=str(data), epochs=epochs, patience=15,
        imgsz=640 if gpu else 480, batch=16 if gpu else 8,
        device=0 if gpu else "cpu", cache=True,
        workers=2 if sys.platform.startswith("win") else 4,
        project=str(HERE / "runs"), name="feeder",
    )
    return Path(model.trainer.save_dir) / "weights" / "best.pt"


def open_camera(preferred):
    import cv2

    for idx in dict.fromkeys([preferred, 0, 1, 2, 3]):  # first one that really delivers frames
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)
        if cap.isOpened() and cap.read()[0]:
            print(f"Using camera {idx}")
            return cap
        cap.release()
    sys.exit("No working camera found. Close other apps using it (browser, Zoom) and check permissions.")


def webcam(weights, cam, conf):
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(weights))
    ids = [i for i, n in model.names.items() if n.lower() in WANTED]
    if not ids:
        sys.exit(f"This model has no cat/dog/person classes: {model.names}")
    print("Showing:", [model.names[i] for i in ids])

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
            cv2.imshow("Feeder test - press q to quit", r.plot())
        except cv2.error:
            sys.exit("OpenCV has no window support. Run: pip uninstall -y opencv-python-headless && pip install opencv-python")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", action="store_true", help="unzip the dataset and train first")
    p.add_argument("--zip", help="dataset zip (implies --train); default: cat-dog-person*.zip next to this file")
    p.add_argument("--weights", help="use this trained .pt file instead of the pretrained model")
    p.add_argument("--model", default="yolo11n.pt", help="base model for training (yolo11s.pt is bigger/slower)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--conf", type=float, default=0.5, help="minimum confidence to show a detection")
    p.add_argument("--cam", type=int, default=0)
    a = p.parse_args()

    if a.train or a.zip:
        weights = train(a.zip, a.model, a.epochs)
        print(f"\nTrained model saved at: {weights}\n")
    elif a.weights:
        weights = Path(a.weights)
        if not weights.exists() and (HERE / weights).exists():
            weights = HERE / weights
    else:
        weights = "yolo11n.pt"  # downloaded automatically the first time
    webcam(weights, a.cam, a.conf)


if __name__ == "__main__":
    main()
