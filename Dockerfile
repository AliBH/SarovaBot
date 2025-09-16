FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# نصب پیش‌نیازهای سیستمی (از جمله libgl1 برای حل مشکل)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# نصب پکیج‌های پایتون
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "bot.py"]
