import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# توکن از ENV خونده می‌شود
TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    raise ValueError("BOT_TOKEN در محیط تنظیم نشده!")

mp_face_mesh = mp.solutions.face_mesh

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام 👋 یک عکس بفرست تا گوشواره روی صورتت بیفته.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    img_path = "input.jpg"
    await photo.download_to_drive(img_path)

    # خواندن تصویر
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # پردازش چهره
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            await update.message.reply_text("صورت پیدا نشد ❌")
            return

        h, w, _ = image.shape
        landmarks = results.multi_face_landmarks[0].landmark

        # مختصات گوش راست (landmark حدود گوش)
        ear_point = landmarks[234]  # نقطه‌ای نزدیک گوش
        x, y = int(ear_point.x * w), int(ear_point.y * h)

        # گوشواره PNG
        earring = Image.open("earring.png").convert("RGBA")
        earring = earring.resize((80, 160))  # تغییر اندازه گوشواره

        base = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        base.paste(earring, (x, y), earring)

        output_path = "output.png"
        base.save(output_path)

    await update.message.reply_photo(photo=open(output_path, "rb"))

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
