FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Libraries OpenCV needs inside a slim image
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch: much smaller than the default CUDA build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the pretrained weights into the image so the event works without internet
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY app.py .
COPY static ./static

EXPOSE 8000
# 1 worker (one copy of the model in memory), 4 threads so several phones can connect at once
CMD ["gunicorn", "-b", "0.0.0.0:8000", "-w", "1", "--threads", "4", "--timeout", "120", "app:app"]
