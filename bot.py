import os
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# -----------------------------
# خواندن توکن از محیط
# -----------------------------
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN در محیط تنظیم نشده!")

# -----------------------------
# مسیر داینامیک فایل گوشواره
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
earring_path = os.path.join(BASE_DIR, "earring.png")
if not os.path.exists(earring_path):
    raise FileNotFoundError(f"فایل گوشواره پیدا نشد: {earring_path}")

# -----------------------------
# هندلر استارت
# -----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! یه عکس بفرست تا گوشواره روی گوش اضافه کنم.")

# -----------------------------
# هندلر عکس
# -----------------------------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    photo_path = os.path.join(BASE_DIR, "user_photo.jpg")
    await photo_file.download(photo_path)

    # باز کردن تصویر کاربر و گوشواره
    user_img = Image.open(photo_path).convert("RGBA")
    earring = Image.open(earring_path).convert("RGBA")

    # ساده‌ترین حالت: گوشواره را روی گوش فرضی قرار می‌دهیم
    earring = earring.resize((50, 50))  # اندازه دلخواه
    user_img.paste(earring, (100, 100), earring)  # جایگذاری روی تصویر

    output_path = os.pa_
