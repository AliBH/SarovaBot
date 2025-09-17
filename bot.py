"""
ربات تلگرام: پرو کردن گوشواره و گردنبند روی عکس کاربر
قابلیت‌ها:
- فهرست محصولات با کدها (گوشواره / گردنبند)
- انتخاب محصول توسط دکمه‌ها یا ارسال کد
- دریافت عکس، تشخیص صورت با Mediapipe، چسبوندن PNG محصول با مقیاس مناسب
- خروجی: عکس PNG به کاربر برگشت داده می‌شود
"""

import os
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import cv2
import mediapipe as mp
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ---------- تنظیمات ----------
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN در محیط تنظیم نشده! (از BotFather بگیر و به ENV اضافه کن)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_DIR = os.path.join(BASE_DIR, "products")

# تعریف کاتالوگ محصولات — اینجا می‌تونی محصولات رو اضافه/ویرایش کنی
# هر محصول: code, name, type ('earring' یا 'necklace'), filename (در پوشه products)
PRODUCTS = {
    "E001": {"name": "گوشواره آویز طلایی", "type": "earring", "file": "E001_earring.png"},
    "E002": {"name": "گوشواره نگین‌دار", "type": "earring", "file": "E002_earring.png"},
    "N001": {"name": "گردنبند ساده", "type": "necklace", "file": "N001_necklace.png"},
    "N002": {"name": "گردنبند مروارید", "type": "necklace", "file": "N002_necklace.png"},
}

# نگهداری حالت هر کاربر: selected product code و آخرین عکس آپلود شده
user_state = {}  # user_id -> {"code": "E001", "last_photo_path": "...", ...}

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# ---------- Utility ----------
def product_filepath(code):
    p = PRODUCTS.get(code)
    if not p:
        return None
    return os.path.join(PRODUCTS_DIR, p["file"])

def list_products_keyboard():
    # دکمه‌های inline از محصولات می‌سازد
    rows = []
    # مرتب‌سازی براساس کد
    for code in sorted(PRODUCTS.keys()):
        p = PRODUCTS[code]
        rows.append([InlineKeyboardButton(f"{code} — {p['name']}", callback_data=f"select:{code}")])
    return InlineKeyboardMarkup(rows)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "سلام! 👋\n"
        "من ربات پرو زیورآلات هستم.\n"
        "برای دیدن کاتالوگ /catalog را بزن.\n"
        "برای انتخاب دستی یک کد، پیام شامل کد (مثلاً E001) ارسال کن.\n"
        "بعد از انتخاب محصول، یک عکس بفرست تا پرو کنم."
    )
    # دکمه شورت‌کات کاتالوگ
    keyboard = ReplyKeyboardMarkup([[KeyboardButton("/catalog")]], resize_keyboard=True)
    await update.message.reply_text(txt, reply_markup=keyboard)

async def catalog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("کاتالوگ محصولات — روی یکی کلیک کن:", reply_markup=list_products_keyboard())

async def handle_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data  # format: "select:CODE"
    if not data.startswith("select:"):
        return
    code = data.split(":", 1)[1].strip().upper()
    if code not in PRODUCTS:
        await query.edit_message_text("کد نامعتبر است.")
        return
    user_id = query.from_user.id
    user_state[user_id] = user_state.get(user_id, {})
    user_state[user_id]["code"] = code
    await query.edit_message_text(f"✅ محصول انتخاب شد: {code} — {PRODUCTS[code]['name']}\nحالا یک عکس بفرست تا پرو کنم.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # اگر کاربر کد محصول فرستاد
    text = (update.message.text or "").strip().upper()
    if text in PRODUCTS:
        user_id = update.message.from_user.id
        user_state[user_id] = user_state.get(user_id, {})
        user_state[user_id]["code"] = text
        await update.message.reply_text(f"✅ محصول انتخاب شد: {text} — {PRODUCTS[text]['name']}\nحالا یک عکس بفرست تا پرو کنم.")
    else:
        await update.message.reply_text("کد محصول پیدا نشد. برای دیدن کاتالوگ /catalog را بزن.")

# ---------- Core: پرو کردن تصویر ----------
def paste_transparent(base_pil: Image.Image, overlay_pil: Image.Image, pos):
    """پیوست شفاف: overlay را در pos (x,y) روی base می‌چسباند"""
    base_pil.paste(overlay_pil, pos, overlay_pil)

def compute_scale_and_position_for_ear(image_w, image_h, landmark, scale_ratio=0.08):
    """
    محاسبه عرض/ارتفاع و مکان بر اساس نقطه گوش (landmark)
    scale_ratio: نسبت عرض گوشواره به عرض تصویر (قابل تنظیم)
    """
    x = int(landmark.x * image_w)
    y = int(landmark.y * image_h)
    ear_w = int(image_w * scale_ratio)  # نسبت به عرض کل تصویر
    ear_h = int(ear_w * 2)  # نسبت عرض به ارتفاع — قابل تنظیم
    return (x - ear_w // 2, y - ear_h // 2, ear_w, ear_h)

def compute_scale_and_position_for_necklace(image_w, image_h, landmarks):
    """
    برای گردنبند: از نقاط چانه (chin) و شانه‌ها تقریبی استفاده می‌کنیم.
    Mediapipe face mesh چین پوینت‌ها در محدوده chin: از 152 و 10 و ... می‌شه استفاده کرد.
    این تابع یک موقعیت مرکزی و عرض برای گردنبند برمی‌گرداند.
    """
    # از چند نقطه‌ی دور چانه میانگین می‌گیریم
    chin_ids = [152, 148, 176, 149]  # نقاط متداول اطراف چانه
    xs, ys = [], []
    for i in chin_ids:
        lm = landmarks.landmark[i]
        xs.append(lm.x)
        ys.append(lm.y)
    cx = int(np.mean(xs) * image_w)
    cy = int(np.mean(ys) * image_h)

    # فاصله از گوش تا گوش (افقی) برای تعیین پهنای گردنبند
    left_ear = landmarks.landmark[234]
    right_ear = landmarks.landmark[454]
    width = int(abs(right_ear.x - left_ear.x) * image_w * 1.2)  # کمی بزرگ‌تر از فاصله گوش‌ها
    height = int(width * 0.4)  # نسبت ارتفاع به عرض
    # قرار دادن گردنبند کمی پایین‌تر از چانه (offset)
    pos_x = cx - width // 2
    pos_y = cy - height // 3  # تنظیم به سمت بالا/پایین قابل تغییر است
    return pos_x, pos_y, width, height

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    state = user_state.get(user_id, {})
    code = state.get("code")
    if not code:
        await update.message.reply_text("اول یک محصول انتخاب کن — /catalog یا کد محصول را ارسال کن.")
        return
    if code not in PRODUCTS:
        await update.message.reply_text("کد محصول نامعتبر است. /catalog را بزن.")
        return

    product = PRODUCTS[code]
    product_file = product_filepath(code)
    if not product_file or not os.path.exists(product_file):
        await update.message.reply_text("فایل محصول روی سرور پیدا نشد — لطفاً ادمین را خبر کن.")
        return

    # دریافت عکس کاربر (bytes)
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    image = np.array(Image.open(BytesIO(photo_bytes)).convert("RGB"))
    h, w, _ = image.shape

    # پردازش face mesh
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        await update.message.reply_text("صورت پیدا نشد — لطفاً یک عکس واضح از صورت بفرست.")
        return

    # بارگذاری تصویر محصول
    overlay = Image.open(product_file).convert("RGBA")

    pil_image = Image.fromarray(image).convert("RGBA")

    for face_landmarks in results.multi_face_landmarks:
        # گوشواره
        if product["type"] == "earring":
            # هر دو گوش: نقاط 234 (left) و 454 (right)
            for ear_landmark_id in (234, 454):
                lm = face_landmarks.landmark[ear_landmark_id]
                x, y, ow, oh = compute_scale_and_position_for_ear(w, h, lm, scale_ratio=0.06)
                if ow <= 0 or oh <= 0:
                    continue
                # تغییر سایز overlay با حفظ نسبت
                ov = overlay.resize((ow, oh), Image.LANCZOS)
                paste_transparent(pil_image, ov, (x, y))
        # گردنبند
        elif product["type"] == "necklace":
            pos_x, pos_y, ow, oh = compute_scale_and_position_for_necklace(w, h, face_landmarks)
            if ow <= 0 or oh <= 0:
                continue
            ov = overlay.resize((ow, oh), Image.LANCZOS)
            paste_transparent(pil_image, ov, (pos_x, pos_y))

    # خروجی به صورت بافر و ارسال
    output = BytesIO()
    output.name = "result.png"
    pil_image.save(output, format="PNG")
    output.seek(0)
    await update.message.reply_photo(photo=output, caption=f"پرو شده: {code} — {product['name']}")

async def show_my_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    state = user_state.get(user_id, {})
    code = state.get("code")
    if not code:
        await update.message.reply_text("هیچ محصولی انتخاب نکردی. /catalog را بزن.")
    else:
        p = PRODUCTS.get(code)
        await update.message.reply_text(f"محصول انتخاب‌شده: {code} — {p['name']} ({p['type']})")

# ---------- Main ----------
def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("catalog", catalog))
    app.add_handler(CommandHandler("my", show_my_selection))
    app.add_handler(CallbackQueryHandler(handle_selection_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    print("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
