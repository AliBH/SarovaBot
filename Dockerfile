FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# نصب پیش‌نیازهای سیستم
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# نصب پکیج‌ها (فقط نسخه headless از OpenCV)
RUN pip install --no-cache-dir python-telegram-bot==20.3 mediapipe opencv-python-headless Pillow numpy

CMD ["python", "bot.py"]
