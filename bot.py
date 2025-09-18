"""
ربات پرو اکسسوری زنانه — نسخه حرفه‌ای کامل
نیازمندی‌ها:
- python-telegram-bot v20+
- mediapipe
- opencv-python-headless
- pillow
- numpy

نحوه افزودن محصول:
- ادمین عکس محصول را ارسال کند با کپشن:
  ADD:<CODE>:<TYPE>:<DISPLAY_NAME>
  مثال: ADD:E001:earring:گوشواره-طلایی

- حذف: /remove <CODE>
- لیست: /list
"""

import os
import logging
from io import BytesIO
from typing import Optional, Tuple, List
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2
import mediapipe as mp
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# -------- config --------
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
ADMIN_IDS_ENV = os.environ.get("ADMIN_IDS", "")  # example: "1234567,7654321"
ADMIN_IDS = {int(x) for x in ADMIN_IDS_ENV.split(",") if x.strip().isdigit()}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_DIR = os.path.join(BASE_DIR, "products")
EARRINGS_DIR = os.path.join(PRODUCTS_DIR, "earrings")
NECKLACES_DIR = os.path.join(PRODUCTS_DIR, "necklaces")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SHOP_BG = os.path.join(ASSETS_DIR, "shop_mirror.jpg")  # تصویر پس‌زمینه مغازه/آینه

# ensure folders exist
os.makedirs(EARRINGS_DIR, exist_ok=True)
os.makedirs(NECKLACES_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN تنظیم نشده!")

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_seg = mp.solutions.selfie_segmentation

# instantiate on demand for thread-safety
def get_face_mesh():
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def get_selfie_seg():
    return mp_selfie_seg.SelfieSegmentation(model_selection=1)

# ---------- helpers ----------

def list_products() -> List[Tuple[str,str,str]]:
    """Return list of (code, type, filename)"""
    items = []
    # earrings
    for fname in sorted(os.listdir(EARRINGS_DIR)):
        if fname.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
            code = os.path.splitext(fname)[0]
            items.append((code, "earring", os.path.join(EARRINGS_DIR, fname)))
    # necklaces
    for fname in sorted(os.listdir(NECKLACES_DIR)):
        if fname.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
            code = os.path.splitext(fname)[0]
            items.append((code, "necklace", os.path.join(NECKLACES_DIR, fname)))
    return items

def product_path_by_code(code: str) -> Optional[Tuple[str,str]]:
    """Return (type, path) for code or None"""
    p = os.path.join(EARRINGS_DIR, code + ".png")
    if os.path.exists(p):
        return ("earring", p)
    p = os.path.join(NECKLACES_DIR, code + ".png")
    if os.path.exists(p):
        return ("necklace", p)
    # also allow jpg/jpeg
    for ext in ("jpg","jpeg","webp","png"):
        p = os.path.join(EARRINGS_DIR, f"{code}.{ext}")
        if os.path.exists(p):
            return ("earring", p)
        p = os.path.join(NECKLACES_DIR, f"{code}.{ext}")
        if os.path.exists(p):
            return ("necklace", p)
    return None

def open_image_from_bytes(b: BytesIO) -> Image.Image:
    b.seek(0)
    return Image.open(b).convert("RGBA")

def save_bytes_to_product(b: BytesIO, code: str, typ: str, display_name: Optional[str]=None) -> str:
    """
    Save uploaded product bytes into products folder.
    typ in {"earring","necklace"}
    returns saved filepath
    """
    b.seek(0)
    ext = ".png"
    # try to detect format from PIL
    try:
        im = Image.open(b)
        fmt = im.format.lower()
        if fmt in ("png","jpeg","jpg","webp"):
            if fmt == "jpeg":
                ext = ".jpg"
            else:
                ext = "." + fmt
        im.close()
    except Exception:
        ext = ".png"
    b.seek(0)
    if typ == "earring":
        dst = os.path.join(EARRINGS_DIR, f"{code}{ext}")
    else:
        dst = os.path.join(NECKLACES_DIR, f"{code}{ext}")
    with open(dst, "wb") as f:
        f.write(b.read())
    return dst

# ---------- image processing ----------

def remove_background_pil(img: Image.Image, bg_img: Optional[Image.Image]=None) -> Image.Image:
    """
    Remove background using MediaPipe Selfie Segmentation and composite over bg_img (PIL).
    If bg_img is None, make transparent background.
    """
    img_rgb = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    with get_selfie_seg() as seg:
        results = seg.process(img_rgb)
    if results.segmentation_mask is None:
        # fallback: return original
        return img
    mask = results.segmentation_mask
    mask = (mask > 0.5).astype(np.uint8) * 255  # 0 or 255
    mask = Image.fromarray(mask).convert("L").resize(img.size, resample=Image.BILINEAR)
    fg = img.convert("RGBA")
    if bg_img is None:
        # keep transparent background
        out = Image.new("RGBA", img.size)
        out.paste(fg, (0,0), mask)
        return out
    else:
        bg = bg_img.convert("RGBA").resize(img.size, Image.LANCZOS)
        out = Image.composite(fg, bg, mask)
        return out

def enhance_image_smart(img: Image.Image) -> Image.Image:
    """
    Smart enhancement:
    - apply CLAHE on L channel
    - adjust contrast/brightness slightly with PIL
    - apply a mild unsharp mask (optional)
    """
    # to OpenCV
    rgb = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    # mild enhancements
    pil = ImageEnhance.Color(pil).enhance(1.05)
    pil = ImageEnhance.Contrast(pil).enhance(1.08)
    pil = ImageEnhance.Brightness(pil).enhance(1.03)
    return pil.convert("RGBA")

def compute_face_angle_and_scale(landmarks, img_w, img_h):
    """Return angle (deg) and scale basis using eye centers and ear distance"""
    # eye landmarks: 33 (left eye outer) and 263 (right eye outer)
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    lx, ly = left_eye.x*img_w, left_eye.y*img_h
    rx, ry = right_eye.x*img_w, right_eye.y*img_h
    dx, dy = rx - lx, ry - ly
    angle = np.degrees(np.arctan2(dy, dx))
    # ear markers for width
    # use approximate ear-like landmarks (234 & 454)
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    ear_dist = np.hypot((right_ear.x-left_ear.x)*img_w, (right_ear.y-left_ear.y)*img_h)
    return angle, max(ear_dist, 1.0)

def get_precise_ear_centers(landmarks, img_w, img_h):
    """
    use multiple landmarks to compute center of lobe:
    left: average of [234, 93, 132]
    right: average of [454, 263, 361]
    """
    left_ids = [234, 93, 132]
    right_ids = [454, 263, 361]
    lx = np.mean([landmarks[i].x for i in left_ids]) * img_w
    ly = np.mean([landmarks[i].y for i in left_ids]) * img_h
    rx = np.mean([landmarks[i].x for i in right_ids]) * img_w
    ry = np.mean([landmarks[i].y for i in right_ids]) * img_h
    return (int(lx), int(ly)), (int(rx), int(ry))

def get_neck_position_and_width(landmarks, img_w, img_h):
    """
    Estimate neck center & width using chin (152) and jaw points (234,454)
    returns (x,y,width)
    """
    chin = landmarks[152]
    jaw_l = landmarks[234]
    jaw_r = landmarks[454]
    cx = int(chin.x * img_w)
    cy = int(chin.y * img_h) + int(0.08 * img_h)  # a bit lower than chin
    width = int(abs(jaw_r.x - jaw_l.x) * img_w * 1.15)
    return (cx, cy, max(width, 40))

def paste_rotated_overlay(base: Image.Image, overlay: Image.Image, center: Tuple[int,int], angle: float, box_size: Tuple[int,int]):
    """
    Rotate overlay by angle (degrees) around its center, resize to box_size, then paste onto base at position so that overlay center maps to center.
    """
    overlay_resized = overlay.resize(box_size, Image.LANCZOS)
    # rotate around center
    overlay_rot = overlay_resized.rotate(angle, expand=True, resample=Image.BICUBIC)
    ow, oh = overlay_rot.size
    cx, cy = center
    paste_pos = (int(cx - ow/2), int(cy - oh/2))
    base.paste(overlay_rot, paste_pos, overlay_rot)
    return base

# ---------- Telegram handlers ----------

def build_main_menu():
    kb = [
        [InlineKeyboardButton("گوشواره", callback_data="cat:earring")],
        [InlineKeyboardButton("گردنبند", callback_data="cat:necklace")],
        [InlineKeyboardButton("لیست محصولات", callback_data="list:all")],
    ]
    return InlineKeyboardMarkup(kb)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام 👋\nبرای پرو، محصول را انتخاب کن یا /help را نگاه کن.", reply_markup=build_main_menu())

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "راهنما:\n"
        "- برای افزودن محصول: ادمین عکس را با کپشن `ADD:<CODE>:<TYPE>:<NAME>` ارسال کند (مثال: ADD:E001:earring:گوشواره-طلایی)\n"
        "- برای حذف محصول: `/remove <CODE>`\n"
        "- فهرست محصولات: /list\n"
        "- برای پرو: محصول را انتخاب کن و سپس یک عکس ارسال کن.\n"
    )
    await update.message.reply_text(txt)

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith("cat:"):
        cat = data.split(":",1)[1]
        # build products list for that category
        buttons = []
        items = list_products()
        for code, typ, path in items:
            if typ == cat:
                buttons.append([InlineKeyboardButton(f"{code}", callback_data=f"prod:{code}")])
        if not buttons:
            await query.edit_message_text("محصولی در این دسته وجود ندارد.")
            return
        await query.edit_message_text(f"محصولات {cat}:", reply_markup=InlineKeyboardMarkup(buttons))
    elif data.startswith("prod:"):
        code = data.split(":",1)[1]
        # set selected product in user_data
        context.user_data["selected_product"] = code
        await query.edit_message_text(f"محصول {code} انتخاب شد. حالا یک عکس بفرست تا پرو کنم.")
    elif data == "list:all":
        items = list_products()
        if not items:
            await query.edit_message_text("هیچ محصولی اضافه نشده است.")
            return
        txt = "فهرست محصولات:\n" + "\n".join([f"{c} ({t})" for c,t,_ in [(i[0],i[1]) for i in items]])
        await query.edit_message_text(txt)

async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    items = list_products()
    if not items:
        await update.message.reply_text("هیچ محصولی وجود ندارد.")
        return
    lines = [f"{code} - {typ} - {os.path.basename(path)}" for code,typ,path in items]
    await update.message.reply_text("محصولات:\n" + "\n".join(lines))

async def remove_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # /remove CODE
    user = update.effective_user
    if ADMIN_IDS and user.id not in ADMIN_IDS:
        await update.message.reply_text("فقط ادمین می‌تواند حذف کند.")
        return
    args = context.args
    if not args:
        await update.message.reply_text("استفاده: /remove <CODE>")
        return
    code = args[0].strip()
    found = False
    for base_dir in (EARRINGS_DIR, NECKLACES_DIR):
        for ext in (".png",".jpg",".jpeg",".webp"):
            p = os.path.join(base_dir, code+ext)
            if os.path.exists(p):
                os.remove(p)
                found = True
    if found:
        await update.message.reply_text(f"{code} حذف شد.")
    else:
        await update.message.reply_text(f"محصول {code} پیدا نشد.")

async def add_product_via_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin uploads image with caption:
    ADD:<CODE>:<TYPE>:<DISPLAY_NAME>
    """
    user = update.effective_user
    if ADMIN_IDS and user.id not in ADMIN_IDS:
        await update.message.reply_text("فقط ادمین می‌تواند محصول اضافه کند.")
        return

    if not update.message.photo:
        await update.message.reply_text("لطفا یک عکس محصول ارسال کن با کپشن ADD:...")
        return

    caption = (update.message.caption or "").strip()
    if not caption.upper().startswith("ADD:"):
        await update.message.reply_text("کپشن فرمت صحیح ندارد. مثال: ADD:E001:earring:گوشواره-طلایی")
        return

    parts = caption.split(":",3)
    if len(parts) < 3:
        await update.message.reply_text("فرمت کپشن اشتباهه. مثال: ADD:E001:earring:گوشواره-طلایی")
        return
    _, code, typ = parts[:3]
    code = code.strip()
    typ = typ.strip().lower()
    display_name = parts[3].strip() if len(parts) >=4 else code

    if typ not in ("earring","necklace"):
        await update.message.reply_text("نوع باید 'earring' یا 'necklace' باشد.")
        return

    # download highest-res photo
    photo = update.message.photo[-1]
    file = await photo.get_file()
    bio = BytesIO()
    await file.download(out=bio)
    saved = save_bytes_to_product(bio, code, typ, display_name)
    await update.message.reply_text(f"محصول ذخیره شد: {saved}")

# ---------- core processing when user sends photo ----------
async def handle_user_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # check selected product
    user = update.effective_user
    sel = context.user_data.get("selected_product")
    if not sel:
        await update.message.reply_text("ابتدا از منو محصول را انتخاب کنید.")
        return

    prod = product_path_by_code(sel)
    if not prod:
        await update.message.reply_text("فایل محصول روی سرور پیدا نشد — لطفاً ادمین را خبر کن.")
        return
    prod_type, prod_path = prod

    # download user photo
    photo = update.message.photo[-1]
    file = await photo.get_file()
    bio = BytesIO()
    await file.download(out=bio)
    bio.seek(0)
    try:
        user_img = open_image_from_bytes(bio)  # RGBA
    except Exception as e:
        logger.exception("cannot open user image")
        await update.message.reply_text("خطا در باز کردن عکس. لطفاً عکس دیگری ارسال کن.")
        return

    # --- 1) remove background and composite over shop mirror ---
    # load shop bg if exists
    bg_img = None
    if os.path.exists(SHOP_BG):
        try:
            bg_img = Image.open(SHOP_BG).convert("RGBA")
        except Exception:
            bg_img = None
    try:
        composited = remove_background_pil(user_img, bg_img)
    except Exception:
        logger.exception("bg removal failed")
        composited = user_img

    # --- 2) enhance image ---
    try:
        enhanced = enhance_image_smart(composited)
    except Exception:
        enhanced = composited

    # convert to numpy BGR for mediapipe
    np_img = cv2.cvtColor(np.array(enhanced.convert("RGB")), cv2.COLOR_RGB2BGR)
    face_mesh = get_face_mesh()
    results = face_mesh.process(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        await update.message.reply_text("صورت پیدا نشد. لطفاً عکس واضح‌تری بفرست.")
        return
    landmarks = results.multi_face_landmarks[0].landmark
    img_h, img_w = np_img.shape[:2]

    # compute geometry: angle & ear centers & neck pos
    angle, ear_dist = compute_face_angle_and_scale(landmarks, img_w, img_h)
    left_center, right_center = get_precise_ear_centers(landmarks, img_w, img_h)
    neck_x, neck_y, neck_w = get_neck_position_and_width(landmarks, img_w, img_h)

    # load product image
    try:
        product_im = Image.open(prod_path).convert("RGBA")
    except Exception:
        await update.message.reply_text("خطا در بارگذاری فایل محصول.")
        return

    # create final canvas (use enhanced as base)
    base = enhanced.convert("RGBA")

    # apply depending on type
    try:
        if prod_type == "earring":
            # determine size based on ear_dist
            # desired earring width proportion: ~0.22 * ear_dist (tweakable)
            e_width = int(max(24, ear_dist * 0.22))
            e_height = int(e_width * product_im.height / product_im.width)
            box = (e_width, e_height)
            # left ear - if visible on image bounds
            for center in (left_center, right_center):
                cx, cy = center
                # if center outside image, skip
                if not (0 <= cx < img_w and 0 <= cy < img_h):
                    continue
                base = paste_rotated_overlay(base, product_im, (cx, cy), angle, box)
        else:  # necklace
            n_w = int(neck_w)
            n_h = int(n_w * product_im.height / max(1, product_im.width))
            box = (n_w, n_h)
            base = paste_rotated_overlay(base, product_im, (neck_x, neck_y), angle, box)
    except Exception:
        logger.exception("error applying product")

    # prepare output
    out_buf = BytesIO()
    base.save(out_buf, "PNG")
    out_buf.seek(0)
    await update.message.reply_photo(photo=out_buf, caption=f"پرو شده: {sel}")

# ---------- set up application ----------

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("remove", remove_cmd))
    # admin adds product by uploading image with caption ADD:...
    app.add_handler(MessageHandler(filters.PHOTO & filters.CaptionRegex(r'(?i)^ADD:'), add_product_via_upload))
    # inline menu callbacks
    app.add_handler(CallbackQueryHandler(callback_handler))
    # user's product selection via callback is handled in callback_handler, now photo handler:
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.CaptionRegex(r'(?i)^ADD:'), handle_user_photo))
    # also text messages might be commands or plain text - we keep minimal
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u,c: c.bot.send_message(u.effective_chat.id, "از منو انتخاب کن یا /help را ببین.")))
    logger.info("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
