import os
import logging
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- تنظیم محصولات ----------------
PRODUCTS = {
    "E001": {"name": "گوشواره طلایی", "file": "products/earring.png", "type": "earring"},
    "E002": {"name": "گوشواره نقره‌ای", "file": "products/earring2.png", "type": "earring"},
    "N001": {"name": "گردنبند ساده", "file": "products/necklace.png", "type": "necklace"},
    "N002": {"name": "گردنبند مروارید", "file": "products/necklace2.png", "type": "necklace"},
}

user_selected_product = {}

# ---------------- Mediapipe ----------------
mp_face_mesh = mp.solutions.face_mesh

def overlay_earrings(user_image_path, product_file):
    img = cv2.imread(user_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    product = Image.open(product_file).convert("RGBA")

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None

        h, w, _ = img.shape
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = (int(landmarks[234].x * w), int(landmarks[234].y * h))
        right_ear = (int(landmarks[454].x * w), int(landmarks[454].y * h))

        ear_distance = np.linalg.norm(np.array(left_ear) - np.array(right_ear))
        resized = product.resize((int(ear_distance*0.25), int(ear_distance*0.5)))

        for ear in [left_ear, right_ear]:
            x, y = ear
            pos = (x - resized.width//2, y)
            base.alpha_composite(resized, dest=pos)

    return base

def overlay_necklace(user_image_path, product_file):
    img = cv2.imread(user_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    product = Image.open(product_file).convert("RGBA")

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None

        h, w, _ = img.shape
        landmarks = results.multi_face_landmarks[0].landmark
        chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))

        width = int(w * 0.6)
        resized = product.resize((width, int(width * product.height / product.width)))

        pos = (w//2 - resized.width//2, chin[1])
        base.alpha_composite(resized, dest=pos)

    return base

# ---------------- Telegram Bot ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["E001", "E002"], ["N001", "N002"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "سلام! 👋\nکد محصول مورد نظر رو انتخاب کن (مثلاً E001 یا N001):",
        reply_markup=reply_markup
    )

async def handle_product_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    code = update.message.text.strip()
    if code in PRODUCTS:
        user_selected_product[update.effective_chat.id] = code
        await update.message.reply_text(f"✅ محصول {PRODUCTS[code]['name']} انتخاب شد.\nحالا عکست رو بفرست 📸")
    else:
        await update.message.reply_text("❌ کد محصول معتبر نیست. دوباره انتخاب کن.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    if user_id not in user_selected_product:
        await update.message.reply_text("اول باید کد محصول انتخاب کنی.")
        return

    photo = update.message.photo[-1]
    file = await photo.get_file()
    img_path = f"user_{user_id}.jpg"
    await file.download_to_drive(img_path)

    code = user_selected_product[user_id]
    product = PRODUCTS[code]
    product_path = product["file"]

    if not os.path.exists(product_path):
        await update.message.reply_text("⚠️ فایل محصول روی سرور پیدا نشد — لطفاً ادمین را خبر کن.")
        return

    if product["type"] == "earring":
        result = overlay_earrings(img_path, product_path)
    else:
        result = overlay_necklace(img_path, product_path)

    if result:
        output_path = f"output_{user_id}.png"
        result.save(output_path)
        await update.message.reply_photo(photo=open(output_path, "rb"))
        os.remove(output_path)
    else:
        await update.message.reply_text("❌ صورت شناسایی نشد. عکس واضح‌تر بفرست.")

if __name__ == "__main__":
    TOKEN = os.getenv("BOT_TOKEN")
    if not TOKEN:
        raise ValueError("BOT_TOKEN در محیط تنظیم نشده!")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_product_code))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Bot is running...")
    app.run_polling()
