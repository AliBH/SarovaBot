import os
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# توکن رو از متغیر محیطی Railway می‌خونه
BOT_TOKEN = os.getenv("BOT_TOKEN")

earring_image = Image.open("earring.png").convert("RGBA")
user_state = {}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام 👋\nعکس خودت رو بفرست تا گوشواره روی گوش‌هات اضافه کنم.\n"
        "حالت پیش‌فرض: فقط گوش راست ✅"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_state[user_id] = "right"

    file = await update.message.photo[-1].get_file()
    photo_path = f"{user_id}_photo.jpg"
    await file.download_to_drive(photo_path)

    processed_image_path = process_image(photo_path, user_state[user_id])

    keyboard = [[InlineKeyboardButton("🔄 یک گوش / دو گوش", callback_data="toggle")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_photo(photo=open(processed_image_path, "rb"), reply_markup=reply_markup)

def process_image(image_path, mode="right"):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    pil_image = Image.fromarray(rgb_image).convert("RGBA")

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            img_h, img_w, _ = image.shape

            # مختصات تقریبی گوش‌ها
            right_x = int(face_landmarks.landmark[454].x * img_w)
            right_y = int(face_landmarks.landmark[454].y * img_h)
            left_x = int(face_landmarks.landmark[234].x * img_w)
            left_y = int(face_landmarks.landmark[234].y * img_h)

            # محاسبه فاصله دو گوش برای تعیین سایز گوشواره
            ear_distance = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
            scale_factor = int(ear_distance * 0.25)  # حدود 25٪ فاصله گوش‌ها

            resized_earring = earring_image.resize((scale_factor, scale_factor))

            if mode in ["right", "both"]:
                pil_image.paste(resized_earring, (right_x - scale_factor//2, right_y - scale_factor//2), resized_earring)
            if mode in ["left", "both"]:
                pil_image.paste(resized_earring, (left_x - scale_factor//2, left_y - scale_factor//2), resized_earring)

    output_path = image_path.replace(".jpg", "_earring.png")
    pil_image.save(output_path)
    return output
