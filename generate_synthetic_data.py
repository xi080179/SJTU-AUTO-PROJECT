"""
Synthetic training data generator for Lego piece detection.
Generates composite images by pasting template pieces onto backgrounds
with augmentations, then applies the preprocessing pipeline to match
the style of project_image_preprocess/ test images.

Output: YOLO-format dataset (images/ + labels/)
"""
import cv2
import numpy as np
from pathlib import Path
import json
import random

# ── Paths ──────────────────────────────────────────────────────
TEMPLATE_DIR = Path('training_data/multiview_templates')
BACKGROUND_DIR = Path('project_image')
OUTPUT_DIR = Path('training_data/synthetic_dataset')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images' / 'val').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

# ── Load class mapping ─────────────────────────────────────────
with open(TEMPLATE_DIR / '_part_index.json', 'r', encoding='utf-8') as f:
    PART_INDEX = json.load(f)
CLASS_NAMES = [PART_INDEX[str(i).zfill(2)]['name'] for i in range(18)]


# ═══════════════════════════════════════════════════════════════
# Preprocessing (mirrored from preprocess.py)
# ═══════════════════════════════════════════════════════════════

def remove_large_shadows(L_channel, kernel_sigma):
    """Remove large-scale shadows from L channel."""
    bg = cv2.GaussianBlur(L_channel, (0, 0), sigmaX=kernel_sigma, sigmaY=kernel_sigma)
    L_float = L_channel.astype(np.float32)
    bg_float = bg.astype(np.float32)
    corrected = L_float - bg_float
    min_val, max_val = np.min(corrected), np.max(corrected)
    if max_val > min_val:
        corrected = 255 * (corrected - min_val) / (max_val - min_val)
    else:
        corrected = np.zeros_like(corrected)
    return corrected.astype(np.uint8)


def apply_preprocess(img_bgr):
    """Apply the same preprocessing as preprocess.py to match test image style."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = remove_large_shadows(L, kernel_sigma=101)
    L = remove_large_shadows(L, kernel_sigma=51)
    L = remove_large_shadows(L, kernel_sigma=12)
    L = cv2.GaussianBlur(L, (0, 0), sigmaX=5, sigmaY=5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)
    blurred = cv2.GaussianBlur(L_eq, (0, 0), sigmaX=3, sigmaY=3)
    unsharp = cv2.addWeighted(L_eq, 1.5, blurred, -0.5, 0)
    laplacian = cv2.Laplacian(unsharp, cv2.CV_16S, ksize=3)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    L_enhanced = cv2.addWeighted(unsharp, 1.0, laplacian_abs, 0.3, 0)
    L_enhanced = remove_large_shadows(L_enhanced, kernel_sigma=100)
    lab_out = cv2.merge([L_enhanced, A, B])
    img_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 120, 230)
    edges = cv2.bitwise_not(edges)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_out = cv2.addWeighted(img_out, 1.0, edges_3ch, 0.3, 0)
    return img_out


# ═══════════════════════════════════════════════════════════════
# Template loading
# ═══════════════════════════════════════════════════════════════

def create_mask_from_white_bg(bgr):
    """Create alpha mask: separate piece from white (255) background."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def load_templates():
    """Load top-view templates, cropped to piece bounds with alpha mask.
    Returns: list of (class_id, rgba) tuples.
    """
    templates = []
    for idx in range(18):
        matching = list(TEMPLATE_DIR.glob(f'{idx:02d}_*'))
        if not matching:
            continue
        top_dir = matching[0] / 'top'
        for img_path in top_dir.glob('*.png'):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            bgr = img[:, :, :3]
            alpha = create_mask_from_white_bg(bgr)
            # Crop to piece bounds
            ys, xs = np.where(alpha > 128)
            if len(xs) < 10 or len(ys) < 10:
                continue
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            bgr_cropped = bgr[y1:y2, x1:x2]
            alpha_cropped = alpha[y1:y2, x1:x2]
            rgba = cv2.merge([bgr_cropped, alpha_cropped])
            templates.append((idx, rgba))
    return templates


# ═══════════════════════════════════════════════════════════════
# Augmentation & transform
# ═══════════════════════════════════════════════════════════════

def augment_piece(rgba):
    """Random color augmentation. Returns (augmented_rgba, angle)."""
    angle = random.uniform(-30, 30)
    bgr = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]
    brightness = random.uniform(0.85, 1.15)
    contrast = random.uniform(0.85, 1.15)
    bgr = contrast * bgr + (brightness - 1) * 128
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    # HSV shift
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.3), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.merge([bgr, alpha]), angle


def rotate_rgba_expand(rgba, angle):
    """Rotate with expanded canvas so the piece is never cropped."""
    h, w = rgba.shape[:2]
    rad = np.radians(abs(angle))
    new_w = int(w * abs(np.cos(rad)) + h * abs(np.sin(rad)))
    new_h = int(w * abs(np.sin(rad)) + h * abs(np.cos(rad)))
    center = (w // 2, h // 2)
    new_center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += new_center[0] - center[0]
    M[1, 2] += new_center[1] - center[1]
    return cv2.warpAffine(rgba, M, (new_w, new_h),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def get_alpha_bbox(rgba):
    """Bounding box of non-zero alpha pixels in RGBA image."""
    ys, xs = np.where(rgba[:, :, 3] > 64)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ═══════════════════════════════════════════════════════════════
# Compositing
# ═══════════════════════════════════════════════════════════════

def add_drop_shadow(canvas, mask, x, y, pw, ph, offset=4, blur=7):
    """Darken canvas region with blurred mask to simulate shadow."""
    shadow = np.zeros_like(canvas, dtype=np.float32)
    sx = max(0, x + offset)
    sy = max(0, y + offset)
    ex = min(canvas.shape[1], sx + pw)
    ey = min(canvas.shape[0], sy + ph)
    mw, mh = ex - sx, ey - sy
    if mw <= 0 or mh <= 0:
        return canvas
    mask_r = cv2.resize(mask, (mw, mh))
    shadow[sy:ey, sx:ex] = mask_r[:, :, np.newaxis] * 0.45
    shadow = cv2.GaussianBlur(shadow, (blur, blur), 0)
    result = canvas.astype(np.float32) * (1 - shadow / 255)
    return np.clip(result, 0, 255).astype(np.uint8)


def blend_piece(canvas, rgba, px, py):
    """Alpha-blend RGBA piece onto canvas at (px, py).
    Returns (updated_canvas, boolean_mask).
    """
    ph, pw = rgba.shape[:2]
    bgr = rgba[:, :, :3]
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    ex = min(canvas.shape[1], px + pw)
    ey = min(canvas.shape[0], py + ph)
    mw, mh = ex - px, ey - py
    if mw <= 0 or mh <= 0:
        return canvas, np.zeros(canvas.shape[:2], dtype=np.uint8)
    a = alpha[:mh, :mw]
    b = bgr[:mh, :mw]
    c = canvas[py:ey, px:ex].astype(np.float32)
    canvas[py:ey, px:ex] = (b * a[:, :, np.newaxis] + c * (1 - a[:, :, np.newaxis])).astype(np.uint8)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    mask[py:ey, px:ex] = (a * 255).astype(np.uint8)
    return canvas, mask


def random_background_crop(bg_images, size):
    """Random crop from a random background image."""
    img = random.choice(bg_images)
    h, w = img.shape[:2]
    if h <= size or w <= size:
        s = size / min(h, w) + 0.1
        img = cv2.resize(img, None, fx=s, fy=s)
        h, w = img.shape[:2]
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return img[y:y + size, x:x + size]


# ═══════════════════════════════════════════════════════════════
# Single image generation
# ═══════════════════════════════════════════════════════════════

def generate_one(templates, bg_images, canvas_size=960):
    """Generate one synthetic image with 3-15 pieces.
    Returns (preprocessed_image, yolo_annotations) or (None, None).
    """
    canvas = random_background_crop(bg_images, canvas_size).copy()
    n_pieces = random.randint(3, 15)
    annotations = []

    for _ in range(n_pieces):
        cls_id, rgba = random.choice(templates)
        rgba_aug, angle = augment_piece(rgba)
        h, w = rgba_aug.shape[:2]

        # Resize to target dimension range
        target = random.randint(60, 280)
        factor = target / max(w, h)
        nw, nh = int(w * factor), int(h * factor)
        rgba_rs = cv2.resize(rgba_aug, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

        # Rotate with expanded canvas
        rgba_rot = rotate_rgba_expand(rgba_rs, angle)

        # Crop to piece's tight bounding box (remove padding from rotation expansion)
        inner = get_alpha_bbox(rgba_rot)
        if inner is None:
            continue
        ix1, iy1, ix2, iy2 = inner
        roi = rgba_rot[iy1:iy2, ix1:ix2]
        roi_h, roi_w = roi.shape[:2]
        if roi_w < 12 or roi_h < 12 or roi_w > canvas_size * 0.9 or roi_h > canvas_size * 0.9:
            continue

        # Random position on canvas
        margin = 5
        mx = canvas_size - roi_w - margin
        my = canvas_size - roi_h - margin
        if mx <= margin or my <= margin:
            continue
        px = random.randint(margin, mx)
        py = random.randint(margin, my)

        # Shadow + blend
        roi_mask = (roi[:, :, 3] > 64).astype(np.uint8) * 255
        canvas = add_drop_shadow(canvas, roi_mask, px, py, roi_w, roi_h)
        canvas, _ = blend_piece(canvas, roi, px, py)

        # YOLO annotation: class x_center y_center width height (normalized 0-1)
        xc = (px + roi_w / 2) / canvas_size
        yc = (py + roi_h / 2) / canvas_size
        nw_n = roi_w / canvas_size
        nh_n = roi_h / canvas_size
        if nw_n < 0.015 or nh_n < 0.015 or nw_n > 0.95 or nh_n > 0.95:
            continue
        xc = np.clip(xc, 0.0, 1.0)
        yc = np.clip(yc, 0.0, 1.0)
        nw_n = np.clip(nw_n, 0.0, 1.0)
        nh_n = np.clip(nh_n, 0.0, 1.0)
        annotations.append(f'{cls_id} {xc:.6f} {yc:.6f} {nw_n:.6f} {nh_n:.6f}')

    if len(annotations) < 2:
        return None, None

    # Apply preprocessing to match test image style
    preprocessed = apply_preprocess(canvas)
    return preprocessed, annotations


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print('Loading templates...')
    templates = load_templates()
    classes_found = sorted(set(t[0] for t in templates))
    print(f'  {len(templates)} templates from {len(classes_found)}/18 classes')

    print('Loading backgrounds...')
    bg_images = []
    for p in sorted(BACKGROUND_DIR.glob('*.jpg')):
        img = cv2.imread(str(p))
        if img is not None:
            bg_images.append(img)
    print(f'  {len(bg_images)} background images')

    TOTAL = 600
    val_ratio = 0.2
    count = 0
    for i in range(TOTAL):
        img, anns = generate_one(templates, bg_images, canvas_size=640)
        if img is None:
            continue
        split = 'val' if random.random() < val_ratio else 'train'
        name = f'syn_{count:04d}'
        img = cv2.resize(img, (640, 640))
        cv2.imwrite(str(OUTPUT_DIR / 'images' / split / f'{name}.jpg'), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(OUTPUT_DIR / 'labels' / split / f'{name}.txt', 'w') as f:
            f.write('\n'.join(anns) + '\n')
        count += 1
        if count % 100 == 0:
            print(f'  {count} generated...')

    train_imgs = list((OUTPUT_DIR / 'images' / 'train').glob('*.jpg'))
    val_imgs = list((OUTPUT_DIR / 'images' / 'val').glob('*.jpg'))
    print(f'\nDone: {len(train_imgs)} train + {len(val_imgs)} val = {count} images')

    # Write dataset.yaml
    names_str = ', '.join(f'"{n}"' for n in CLASS_NAMES)
    yaml = f"""# YOLOv8 Lego 18-class synthetic dataset
path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
nc: 18
names: [{names_str}]
"""
    yaml_path = OUTPUT_DIR / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml)
    print(f'  dataset.yaml → {yaml_path}')


if __name__ == '__main__':
    main()
