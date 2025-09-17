import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN در محیط تنظیم نشده!")

PRODUCTS_DIR = os.path.join(os.getcwd(), "products")

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

PRODUCTS = {
    "earring1": "earring1.png",
    "earring2": "earring2.png",
    "necklace1": "necklace1.png",
    "necklace2": "necklace2.png"
}

def get_main_menu():
    keyboard = [
        [InlineKeyboardButton("گوشواره", callback_data="category_earring")],
        [InlineKeyboardButton("گردنبند", callback_data="category_necklace")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_products_menu(category):
    buttons = []
    for code, filename in PRODUCTS.items():
        if category in code:
            buttons.append([InlineKeyboardButton(code, callback_data=f"product_{code}")])
    return InlineKeyboardMarkup(buttons)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! محصول مورد نظرت را انتخاب کن:", reply_markup=get_main_menu())

async def handle_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("category_"):
        category = data.split("_")[1]
        await query.edit_message_text(f"محصولات {category}:", reply_markup=get_products_menu(category))
    elif data.startswith("product_"):
        code = data.split("_")[1]
        context.user_data["selected_product"] = code
        await query.edit_message_text(f"شما محصول {code} را انتخاب کردید. لطفا عکس خود را ارسال کنید.")

def enhance_image(image: Image.Image) -> Image.Image:
    img = image.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    return img

def apply_product(user_image_path, product_code):
    if product_code not in PRODUCTS:
        raise FileNotFoundError("کد محصول معتبر نیست!")

    product_path = os.path.join(PRODUCTS_DIR, PRODUCTS[product_code])
    if not os.path.exists(product_path):
        raise FileNotFoundError(f"فایل محصول {product_path} پیدا نشد!")

    user_img = Image.open(user_image_path).convert("RGBA")
    product_img = Image.open(product_path).convert("RGBA")

    img_np = np.array(user_img)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    results = mp_face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return user_img

    face_landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img_rgb.shape

    # نقاط تقریبی گوش‌ها و گردن
    left_ear = face_landmarks[234]
    right_ear = face_landmarks[454]
    chin = face_landmarks[152]

    left_pos = (int(left_ear.x * w), int(left_ear.y * h))
    right_pos = (int(right_ear.x * w), int(right_ear.y * h))
    neck_pos = (int(chin.x * w), int(chin.y * h))

    result_img = user_img.copy()
    # چک نوع محصول
    if "earring" in product_code:
        ear_distance = np.linalg.norm(np.array(left_pos) - np.array(right_pos))
        scale_factor = ear_distance / product_img.width * 0.6
        new_size = (int(product_img.width * scale_factor), int(product_img.height * scale_factor))
        product_img_resized = product_img.resize(new_size, Image.ANTIALIAS)
        # قرار دادن روی هر دو گوش
        for pos in [left_pos, right_pos]:
            result_img.paste(product_img_resized, (pos[0]-new_size[0]//2, pos[1]-new_size[1]//2), product_img_resized)
    else:
        # گردنبند روی گردن
        scale_factor = w / 4 / product_img.width
        new_size = (int(product_img.width * scale_factor), int(product_img.height * scale_factor))
        product_img_resized = product_img.resize(new_size, Image.ANTIALIAS)
        result_img.paste(product_img_resized, (neck_pos[0]-new_size[0]//2, neck_pos[1]-new_size[1]//2), product_img_resized)

    return result_img

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    file_path = f"temp_{update.message.from_user.id}.png"
    await file.download(file_path)

    enhanced_img = enhance_image(Image.open(file_path))
    enhanced_img.save(file_path)

    product_code = context.user_data.get("selected_product")
    if not product_code:
        await update.message.reply_text("لطفا ابتدا محصول را از منو انتخاب کنید.", reply_markup=get_main_menu())
        return

    try:
        result_img = apply_product(file_path, product_code)
        result_path = f"result_{update.message.from_user.id}.png"
        result_img.save(result_path)
        await update.message.reply_photo(photo=open(result_path, "rb"))
    except FileNotFoundError as e:
        await update.message.reply_text(str(e))

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_menu))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
