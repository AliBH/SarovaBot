import os
from io import BytesIO
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import mediapipe as mp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes

# تنظیم مسیر دینامیک فایل‌ها
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_DIR = os.path.join(BASE_DIR, "products")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# محصول نمونه
products = {
    "earring1": "earring1.png",
    "earring2": "earring2.png",
    "necklace1": "necklace1.png"
}

# تنظیم نور و کنتراست خودکار
def auto_enhance(image: Image.Image) -> Image.Image:
    # تبدیل به HSV برای کنترل روشنایی
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)  # بهبود کانال روشنایی
    lab = cv2.merge((l, a, b))
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return image

# قرار دادن گوشواره روی گوش
def place_earring_on_face(face_img: Image.Image, earring_img: Image.Image) -> Image.Image:
    img_cv = np.array(face_img)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return face_img  # اگر چهره پیدا نشد، عکس اصلی برگردد

    face_landmarks = results.multi_face_landmarks[0]

    # شاخص لاله گوش (تقریباً نقاط 234 و 454 برای چپ و راست)
    left_ear = face_landmarks.landmark[234]
    right_ear = face_landmarks.landmark[454]

    h, w, _ = img_cv.shape
    positions = [
        (int(left_ear.x * w), int(left_ear.y * h)),
        (int(right_ear.x * w), int(right_ear.y * h))
    ]

    earring_resized = earring_img.resize((int(w * 0.05), int(w * 0.05)))  # سایز گوشواره متناسب با عکس

    for pos in positions:
        x, y = pos
        ex, ey = earring_resized.size
        box = (x - ex//2, y - ey//2)
        face_img.paste(earring_resized, box, earring_resized)

    return face_img

# ساخت منوی محصولات
def build_product_menu():
    buttons = []
    for code, filename in products.items():
        buttons.append([InlineKeyboardButton(code, callback_data=code)])
    return InlineKeyboardMarkup(buttons)

# دستورات ربات
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! محصول مورد نظر را انتخاب کنید:", reply_markup=build_product_menu())

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    product_code = query.data
    context.user_data['selected_product'] = product_code
    await query.edit_message_text(text=f"شما محصول {product_code} را انتخاب کردید. حالا یک عکس ارسال کنید.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    product_code = context.user_data.get('selected_product')
    if not product_code:
        await update.message.reply_text("لطفاً اول محصول را انتخاب کنید.")
        return

    product_file = os.path.join(PRODUCTS_DIR, products[product_code])
    if not os.path.isfile(product_file):
        await update.message.reply_text("فایل محصول روی سرور پیدا نشد — لطفاً ادمین را خبر کن.")
        return

    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = BytesIO()
    await photo_file.download(out=photo_bytes)
    face_img = Image.open(photo_bytes).convert("RGBA")
    face_img = auto_enhance(face_img)

    product_img = Image.open(product_file).convert("RGBA")
    if "earring" in product_code:
        result_img = place_earring_on_face(face_img, product_img)
    else:
        # TODO: برای گردنبند می‌توان مشابه عمل کرد
        result_img = face_img

    output = BytesIO()
    output.name = "result.png"
    result_img.save(output, "PNG")
    output.seek(0)
    await update.message.reply_photo(output)

# اجرای ربات
TOKEN = os.environ.get("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN در محیط تنظیم نشده!")

app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(button))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()
