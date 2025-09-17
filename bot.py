import os
from io import BytesIO
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import mediapipe as mp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# مسیر داینامیک برای فایل‌ها
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_DIR = os.path.join(BASE_DIR, "products")  # پوشه محصولات

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# محصولات نمونه
PRODUCTS = {
    "earring1": os.path.join(PRODUCTS_DIR, "earring1.png"),
    "earring2": os.path.join(PRODUCTS_DIR, "earring2.png"),
    "necklace1": os.path.join(PRODUCTS_DIR, "necklace1.png"),
    "necklace2": os.path.join(PRODUCTS_DIR, "necklace2.png")
}

# تابع تنظیم خودکار نور و کنتراست
def auto_adjust_image(pil_img):
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.3)
    return pil_img

# تابع اعمال گوشواره روی لاله گوش
def apply_earring(image, earring_path, ear_coords):
    earring = Image.open(earring_path).convert("RGBA")
    x1, y1 = ear_coords[0]
    if len(ear_coords) > 1:
        x2, y2 = ear_coords[1]
        width = int(abs(x2 - x1) * 1.5)
    else:
        width = 50  # گوشواره تک گوش
    earring = earring.resize((width, width))
    image.paste(earring, (x1 - width//2, y1 - width//2), earring)
    return image

# تابع اعمال گردنبند روی گردن
def apply_necklace(image, necklace_path, neck_coords):
    necklace = Image.open(necklace_path).convert("RGBA")
    x, y, w = neck_coords
    necklace = necklace.resize((w, int(w/2)))
    image.paste(necklace, (x, y), necklace)
    return image

# شناسایی لاله گوش با MediaPipe
def get_ear_coords(image):
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = mp_face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0]
    left_ear = (int(face_landmarks.landmark[234].x * image.width),
                int(face_landmarks.landmark[234].y * image.height))
    right_ear = (int(face_landmarks.landmark[454].x * image.width),
                 int(face_landmarks.landmark[454].y * image.height))
    return [left_ear, right_ear]

# شناسایی گردن (تقریبی)
def get_neck_coords(image):
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = mp_face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0]
    x = int(face_landmarks.landmark[152].x * image.width) - 50
    y = int(face_landmarks.landmark[152].y * image.height)
    width = 100
    return (x, y, width)

# منوی انتخاب محصول
def get_products_keyboard():
    buttons = [
        [InlineKeyboardButton("گوشواره 1", callback_data="earring1")],
        [InlineKeyboardButton("گوشواره 2", callback_data="earring2")],
        [InlineKeyboardButton("گردنبند 1", callback_data="necklace1")],
        [InlineKeyboardButton("گردنبند 2", callback_data="necklace2")]
    ]
    return InlineKeyboardMarkup(buttons)

# دستور start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! لطفاً محصول مورد نظر خود را انتخاب کنید:",
        reply_markup=get_products_keyboard()
    )

# انتخاب محصول
async def select_product(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["selected_product"] = query.data
    await query.edit_message_text("لطفاً یک عکس ارسال کنید:")

# دریافت عکس و اعمال محصول
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "selected_product" not in context.user_data:
        await update.message.reply_text("ابتدا محصول را انتخاب کنید.")
        return

    file = await update.message.photo[-1].get_file()
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)
    image = Image.open(bio).convert("RGBA")
    image = auto_adjust_image(image)

    product_code = context.user_data["selected_product"]
    product_path = PRODUCTS.get(product_code)

    if not product_path or not os.path.exists(product_path):
        await update.message.reply_text("فایل محصول روی سرور پیدا نشد — لطفاً ادمین را خبر کن.")
        return

    if "earring" in product_code:
        ear_coords = get_ear_coords(image)
        if not ear_coords:
            await update.message.reply_text("لاله گوش پیدا نشد!")
            return
        image = apply_earring(image, product_path, ear_coords)

    elif "necklace" in product_code:
        neck_coords = get_neck_coords(image)
        if not neck_coords:
            await update.message.reply_text("گردن پیدا نشد!")
            return
        image = apply_necklace(image, product_path, neck_coords)

    out_bio = BytesIO()
    image.save(out_bio, "PNG")
    out_bio.seek(0)
    await update.message.reply_photo(out_bio)

# Main
TOKEN = os.environ.get("BOT_TOKEN")
app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(select_product))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

app.run_polling()
