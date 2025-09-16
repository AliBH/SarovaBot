# از نسخه سبک Python 3.11 استفاده می‌کنیم
FROM python:3.11-slim

# کار در مسیر /app
WORKDIR /app

# نصب کتابخانه‌های لازم سیستم
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# کپی کردن کد و requirements
COPY . /app

# نصب پکیج‌های Python
RUN pip install --no-cache-dir -r requirements.txt

# مشخص کردن دستور پیش‌فرض اجرای بات
CMD ["python", "bot.py"]
