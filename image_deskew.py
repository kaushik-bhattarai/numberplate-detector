import cv2
import numpy as np
import os
from pathlib import Path

INPUT_DIR = "/home/kaushik/Nepali_Num_plate/cropped-plates"
OUTPUT_DIR = "/home/kaushik/Nepali_Num_plate/corrected-plates"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def get_plate_angle_hough(gray):
    """Estimate skew angle using Hough Line Transform."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                             minLineLength=40, maxLineGap=15)
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only keep near-horizontal lines (within ±45°)
        if -45 <= angle <= 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def try_perspective_correction(img, gray):
    """
    Try to find a 4-corner quadrilateral and apply perspective transform.
    Returns corrected image or None if no good quad found.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)

    # Dilate edges slightly to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = img.shape[0] * img.shape[1]

    for c in contours:
        area = cv2.contourArea(c)
        # Contour must cover at least 20% of the image
        if area < 0.20 * img_area:
            break

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            rect = order_points(pts)

            # Compute output width and height
            widthA = np.linalg.norm(rect[1] - rect[0])
            widthB = np.linalg.norm(rect[2] - rect[3])
            heightA = np.linalg.norm(rect[3] - rect[0])
            heightB = np.linalg.norm(rect[2] - rect[1])
            maxW = int(max(widthA, widthB))
            maxH = int(max(heightA, heightB))

            if maxW < 10 or maxH < 10:
                continue

            dst = np.array([
                [0, 0],
                [maxW - 1, 0],
                [maxW - 1, maxH - 1],
                [0, maxH - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, M, (maxW, maxH))
            return warped

    return None

def rotate_image(img, angle):
    """Rotate image by angle degrees without cropping."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated

def correct_plate(img):
    """
    Main correction pipeline:
    1. Try perspective correction (best for warped/angled shots)
    2. Fall back to Hough-based rotation (for simple tilt)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Perspective correction
    corrected = try_perspective_correction(img, gray)
    if corrected is not None:
        return corrected, "perspective"

    # Step 2: Hough-based rotation
    angle = get_plate_angle_hough(gray)
    if abs(angle) < 0.5:
        return img, "no_correction"  # Already straight

    rotated = rotate_image(img, angle)
    return rotated, f"rotated_{angle:.1f}deg"


def process_folder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [
        f for f in Path(INPUT_DIR).iterdir()
        if f.suffix.lower() in SUPPORTED_EXTS
    ]

    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_files)} images. Processing...\n")

    stats = {"perspective": 0, "rotated": 0, "no_correction": 0, "failed": 0}

    for img_path in sorted(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP]  {img_path.name} — could not read")
            stats["failed"] += 1
            continue

        try:
            result, method = correct_plate(img)

            out_path = Path(OUTPUT_DIR) / img_path.name
            cv2.imwrite(str(out_path), result)

            if method.startswith("rotated"):
                stats["rotated"] += 1
            elif method == "perspective":
                stats["perspective"] += 1
            else:
                stats["no_correction"] += 1

            print(f"  [OK]  {img_path.name:40s}  method={method}")

        except Exception as e:
            print(f"  [ERR] {img_path.name} — {e}")
            stats["failed"] += 1

    print(f"""
Done!
  Perspective corrected : {stats['perspective']}
  Rotation corrected    : {stats['rotated']}
  Already straight      : {stats['no_correction']}
  Failed / skipped      : {stats['failed']}
  Output saved to       : {OUTPUT_DIR}
""")

if __name__ == "__main__":
    process_folder()