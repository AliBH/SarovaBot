import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------- تنظیمات ----------------
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("توکن ربات در متغیر محیطی BOT_TOKEN تنظیم نشده!")

PRODUCTS = {
    "E001": "products/earring1.png",
    "E002": "products/earring2.png",
    "N001": "products/necklace1.png",
}

mp_face_mesh = mp.solutions.face_mesh

# ---------------- پردازش تصویر ----------------
def add_product(image_path, product_path, product_type="earring"):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    product = Image.open(product_path).convert("RGBA")

    # فاصله چشم‌ها برای مقیاس‌دهی
    eye_left = landmarks[33]
    eye_right = landmarks[263]
    eye_dist = int(((eye_left.x - eye_right.x) ** 2 + (eye_left.y - eye_right.y) ** 2) ** 0.5 * w)

    if product_type == "earring":
        # نقاط گوش
        left_ear = (int(landmarks[234].x * w), int(landmarks[234].y * h))
        right_ear = (int(landmarks[454].x * w), int(landmarks[454].y * h))

        # سایز گوشواره
        new_size = (eye_dist // 4, eye_dist // 2)
        product = product.resize(new_size)

        # بررسی گوش چپ
        if left_ear[0] > 0 and left_ear[0] < w:
            img_pil.paste(product, (left_ear[0] - new_size[0]//2, left_ear[1]), product)

        # بررسی گوش راست
        if right_ear[0] > 0 and right_ear[0] < w:
            img_pil.paste(product, (right_ear[0] - new_size[0]//2, right_ear[1]), product)

    elif product_type == "necklace":
        chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))
        new_size = (eye_dist, eye_dist // 2)
        product = product.resize(new_size)
        img_pil.paste(product, (chin[0] - new_size[0]//2, chin[1]), product)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------------- هندلرها ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["E001", "E002"], ["N001"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    await update.message.reply_text("سلام! یک کد محصول انتخاب کن:", reply_markup=reply_markup)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    code = update.message.text.strip().upper()
    if code in PRODUCTS:
        context.user_data["product_code"] = code
        await update.message.reply_text(f"محصول {code} انتخاب شد ✅ حالا عکس خودتو بفرست.")
    else:
        await update.message.reply_text("کد محصول معتبر نیست!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "product_code" not in context.user_data:
        await update.message.reply_text("اول یک محصول انتخاب کن (E001, E002, N001).")
        return

    code = context.user_data["product_code"]
    product_path = PRODUCTS.get(code)

    if not os.path.exists(product_path):
        await update.message.reply_text("⚠️ فایل محصول روی سرور پیدا نشد — لطفاً ادمین را خبر کن.")
        return

    # دانلود عکس کاربر
    photo = await update.message.photo[-1].get_file()
    input_path = "input.jpg"
    output_path = "output.jpg"
    await photo.download_to_drive(input_path)

    # تعیین نوع محصول
    product_type = "earring" if code.startswith("E") else "necklace"
    result = add_product(input_path, product_path, product_type)

    if result is None:
        await update.message.reply_text("هیچ صورتی پیدا نشد 😕")
        return

    cv2.imwrite(output_path, result)
    await update.message.reply_photo(photo=open(output_path, "rb"))

# ---------------- اجرای ربات ----------------
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling()

if __name__ == "__main__":
    main()
