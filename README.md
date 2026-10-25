# Automatic Pet Feeder (YOLO11)

> **Status:** the software works: webcam test, image upload, and a web page for events. Hardware integration (Raspberry Pi 3B, dosing mechanism, load cell) is the next phase.
>
> **Current limitation:** the model tells *species* apart (cat × dog × person), not individual animals.

Developed as an Integrator Project (PIE4) at IFSP Salto.

## Preliminary Results

<div align="center">
  <img align="center" alt="Coding" width="100%" height="100%" src="https://github.com/MrNutsu/Automatic-Pet-Feeder/blob/main/preliminary%20results.png"/>
  <p align="right"><a href="https://github.com/MrNutsu" name="Nutsu">mak</a></p>
</div>

## What's inside

| Folder | What it is |
|---|---|
| [`ml-training/`](ml-training) | `hub.py`: desktop window to test with the webcam or an uploaded image. `data_training.py`: unzips the dataset, trains the model and runs the webcam. |
| [`feeder-web/`](feeder-web) | A small web page (Docker) where visitors upload a photo from their phone and see the result. Built for events. |

## Requirements

- **Python 3.10 to 3.12**
- A webcam (only for the webcam test)
- Internet on the first run (the pretrained `yolo11n.pt` model, about 5 MB, downloads automatically)
- Docker (only for the web page)

## Quick start: Linux (Ubuntu, Linux Mint, Debian)

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-tk

git clone https://github.com/MrNutsu/Automatic-Pet-Feeder.git
cd Automatic-pet-Feeder/ml-training

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python hub.py
```

`python3-tk` provides the window toolkit. Skip it **only** if Python already has `tkinter`.

## Quick start: Windows

1. Install **Python 3.10 to 3.12** from [python.org](https://www.python.org/downloads/). During setup, check **"Add python.exe to PATH"**.
2. Get the code: install [Git](https://git-scm.com/download/win) and clone, or use **Code → Download ZIP** on GitHub and unzip it.
3. Open **PowerShell** in the `ml-training` folder and run:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

python hub.py
```

If PowerShell says running scripts is disabled, try running this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Using the test hub

`python hub.py` opens a window with:

- **Model:** the pretrained YOLO11, or your own trained model (available after training, see below).
- **Slider with Minimum confidence accepted:** detections below this value are hidden.
- **Webcam test:** opens the live camera in its own window. Press **q** in that window to close it.
- **Analyze an image:** pick a photo and see the boxes drawn on it, with the classes and confidence.

The camera also works without the window: `python hub.py --webcam` (add `--cam 1` to pick another camera and `--conf 0.6` to change the confidence).

## Training your own model

The training data is **not** included in this repository.

1. Download the dataset from Roboflow Universe: [cat-dog-person](https://universe.roboflow.com/yolotraining-with-some-images/cat-dog-person) (version 4, export format **YOLOv11**). A free Roboflow account is needed to download.
2. Put the `.zip` in the `ml-training` folder, next to `data_training.py`. Its name must start with `cat-dog-person`, or pass `--zip <file>`.
3. Run:

```bash
python data_training.py --train
```

This unzips the dataset, trains, and opens the webcam with the new model. The result is saved in `runs/feeder/weights/best.pt`, and `hub.py` finds it automatically.

Useful options: `--model yolo11s.pt` (bigger and slower), `--epochs 80`, `--conf 0.6`.

> **Speed:** with an NVIDIA GPU, training takes minutes. On a CPU it can take an hour or more, so consider training on Google Colab and copying `best.pt` back.

Webcam only, without training: `python data_training.py` (pretrained model) or `python data_training.py --weights best.pt`.

## Web page for events (Docker)

Visitors open a page on their phone and can upload a photo and see what the model finds. Photos are processed in memory and **never saved**, and uploads are limited to 10 MB.

```bash
cd feeder-web
docker compose up --build
```

- **Linux:** install Docker Engine. If you get *permission denied*, run `sudo usermod -aG docker $USER` and log in again, or use `sudo docker compose up --build`.
- **Windows:** install Docker Desktop.
- The first build takes several minutes and needs internet. After that it works offline.
- Open `http://localhost:8000` on the same computer to check it.
- **On phones:** use `http://<computer-ip>:8000` on the same Wi-Fi. Find the IP with `hostname -I` (Linux) or `ipconfig` (Windows, "IPv4 Address"). Public Wi-Fi often blocks devices from talking to each other, so a phone hotspot or a small router is safer.
- To use your own model, put `best.pt` in `feeder-web/models/` and run `docker compose restart`. Otherwise the pretrained model is used.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Tkinter is missing` | Linux: `sudo apt install python3-tk`. Windows: reinstall Python from python.org. |
| `ModuleNotFoundError` for `ultralytics`, `cv2` or `PIL` | The virtual environment is not active (the prompt should start with `(venv)`). Activate it and run `pip install -r requirements.txt`. |
| `No working camera found` | Close other apps using the camera (browser, Zoom). Try `--cam 1`. On Windows, check Settings → Privacy → Camera. |
| `OpenCV has no window support` | `pip uninstall -y opencv-python-headless` then `pip install opencv-python`. |
| Dogs detected as cats | Raise the confidence slider, or test the pretrained model first. Models trained on small datasets can learn the background instead of the animal. |
| Nothing happens when running a script | Make sure the file is saved. |
| Docker *permission denied* | See the Docker section above. |

## Credits and licenses

- **YOLO11** by Ultralytics: Jocher, G.; Qiu, J. *Ultralytics YOLO11*, 2024. Licensed under AGPL-3.0. <https://github.com/ultralytics/ultralytics>
- **Dataset:** *cat-dog-person* (v4), provided by a Roboflow Universe user, licensed under **CC BY 4.0**. <https://universe.roboflow.com/yolotraining-with-some-images/cat-dog-person>. It is not redistributed here.
- This project uses Ultralytics, which is AGPL-3.0, so this repository is released under the same license. See `LICENSE`.

## Author

- João Ravache / [MrNutsu](https://github.com/MrNutsu)
