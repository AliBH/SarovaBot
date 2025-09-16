# استفاده از Python 3.11 slim
FROM python:3.11-slim

# تنظیم دایرکتوری کاری
WORKDIR /app

# نصب کتابخانه‌های مورد نیاز سیستم برای OpenCV و Mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# کپی کردن تمام فایل‌های پروژه به کانتینر
COPY . /app

# نصب وابستگی‌های پایتون
RUN pip install --no-cache-dir -r requirements.txt

# دستور اجرای بات
CMD ["python", "bot.py"]
