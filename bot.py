import os
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# دریافت توکن از محیط
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN در محیط تنظیم نشده!")

# مسیر دینامیک فایل گوشواره
current_dir = os.path.dirname(os.path.abspath(__file__))
earring_path = os.path.join(current_dir, "earring.png")
if not os.path.exists(earring_path):
    raise FileNotFoundError(f"فایل گوشواره پیدا نشد: {earring_path}")

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! عکس بفرست تا گوشواره بچسبونم 😎")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # دریافت عکس کاربر
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    image = np.array(Image.open(BytesIO(photo_bytes)).convert("RGB"))

    # تشخیص صورت
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        await update.message.reply_text("صورت پیدا نشد 😢")
        return

    earring_img = Image.open(earring_path).convert("RGBA")
    pil_image = Image.fromarray(image)

    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = image.shape
        # مختصات گوش‌ها از Mediapipe (Left & Right)
        left_ear = face_landmarks.landmark[234]   # گوش چپ
        right_ear = face_landmarks.landmark[454]  # گوش راست

        for ear in [left_ear, right_ear]:
            x = int(ear.x * w)
            y = int(ear.y * h)
            # سایز گوشواره نسبت به صورت
            ear_width = int(w * 0.05)
            ear_height = int(h * 0.1)
            earring_resized = earring_img.resize((ear_width, ear_height))
            pil_image.paste(earring_resized, (x - ear_width // 2, y - ear_height // 2), earring_resized)

    # ارسال عکس نهایی
    output = BytesIO()
    output.name = 'result.png'
    pil_image.save(output, 'PNG')
    output.seek(0)
    await update.message.reply_photo(photo=output)

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
