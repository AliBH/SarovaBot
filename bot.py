import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from telegram import ReplyKeyboardMarkup, Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)

# ====== TOKEN ======
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN در محیط تنظیم نشده!")

# ====== Mediapipe ======
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# ====== انتخاب محصول ======
selected_product = {}

# ====== Utility: پیدا کردن دقیق لاله گوش ======
def get_precise_ear_positions(landmarks, img_w, img_h):
    left_points = [landmarks[234], landmarks[93], landmarks[132]]
    lx = int(np.mean([p.x for p in left_points]) * img_w)
    ly = int(np.mean([p.y for p in left_points]) * img_h)

    right_points = [landmarks[454], landmarks[263], landmarks[361]]
    rx = int(np.mean([p.x for p in right_points]) * img_w)
    ry = int(np.mean([p.y for p in right_points]) * img_h)

    return (lx, ly), (rx, ry)

# ====== Utility: پیدا کردن دقیق محل گردنبند ======
def get_neck_position_and_size(landmarks, img_w, img_h):
    chin = landmarks[152]
    jaw_left = landmarks[234]
    jaw_right = landmarks[454]

    cx = int(chin.x * img_w)
    cy = int(chin.y * img_h) + 20  # کمی پایین‌تر از چونه

    neck_width = int(abs(jaw_right.x - jaw_left.x) * img_w * 1.2)  # کمی بزرگتر از عرض فک
    neck_height = int(neck_width * 0.5)  # نسبت ارتفاع به عرض

    return (cx, cy), (neck_width, neck_height)

# ====== Utility: انداختن محصول روی تصویر ======
def overlay_product(user_img, product_path, positions, scale=None, size=None):
    if not os.path.exists(product_path):
        return None

    base = Image.fromarray(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    product = Image.open(product_path).convert("RGBA")

    if size:
        product = product.resize(size)

    for (x, y) in positions:
        if not size and scale:
            new_w = int(product.width * scale)
            new_h = int(product.height * scale)
            product_resized = product.resize((new_w, new_h))
        else:
            product_resized = product

        paste_x = x - product_resized.width // 2
        paste_y = y - product_resized.height // 2
        base.paste(product_resized, (paste_x, paste_y), product_resized)

    return cv2.cvtColor(np.array(base), cv2.COLOR_RGBA2BGR)

# ====== Telegram Handlers ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        ["👂 گوشواره‌ها", "💎 گردنبندها"],
        ["❌ لغو"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("سلام 👋\nلطفاً دسته‌بندی محصول رو انتخاب کن:", reply_markup=reply_markup)

async def handle_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    chat_id = update.message.chat_id

    if text == "👂 گوشواره‌ها":
        selected_product[chat_id] = "earrings"
        await update.message.reply_text("کد گوشواره رو وارد کن (مثلاً E001):")
    elif text == "💎 گردنبندها":
        selected_product[chat_id] = "necklaces"
        await update.message.reply_text("کد گردنبند رو وارد کن (مثلاً N001):")
    elif text == "❌ لغو":
        selected_product.pop(chat_id, None)
        await update.message.reply_text("لغو شد ✅")
    else:
        if chat_id not in selected_product:
            await update.message.reply_text("❗ اول باید دسته‌بندی انتخاب کنی.")
            return
        category = selected_product[chat_id]
        code = text.strip()
        context.user_data["selected_file"] = f"products/{category}/{code}.png"
        await update.message.reply_text(f"محصول {code} انتخاب شد ✅\nحالا یک عکس بفرست!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if "selected_file" not in context.user_data:
        await update.message.reply_text("❗ اول باید محصول رو انتخاب کنی.")
        return

    product_path = context.user_data["selected_file"]
    photo = await update.message.photo[-1].get_file()
    img_path = "input.jpg"
    await photo.download_to_drive(img_path)

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        await update.message.reply_text("❌ صورت پیدا نشد.")
        return

    landmarks = results.multi_face_landmarks[0].landmark

    if "earrings" in product_path:
        positions = get_precise_ear_positions(landmarks, w, h)
        result = overlay_product(img, product_path, positions, scale=0.45)
    else:
        neck_pos, neck_size = get_neck_position_and_size(landmarks, w, h)
        result = overlay_product(img, product_path, [neck_pos], size=neck_size)

    if result is None:
        await update.message.reply_text("❌ فایل محصول روی سرور پیدا نشد — لطفاً ادمین را خبر کن.")
        return

    out_path = "output.png"
    cv2.imwrite(out_path, result)
    await update.message.reply_photo(photo=open(out_path, "rb"))

# ====== Main ======
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
