import os
import cv2
from ultralytics import YOLO

def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def crop_plates_from_video(
    model_path: str,
    video_path: str,
    output_dir: str = None,
    conf: float = 0.6,
    tracker: str = "bytetrack.yaml",
) -> int:
    """
    Detects and tracks number plates in a video, saving the sharpest
    crop per tracked plate.

    Args:
        model_path: Path to the YOLO .pt weights file.
        video_path:  Path to the input video.
        output_dir:  Folder to save cropped plates.
                     Defaults to cropped-detections/<video_name>.
        conf:        Minimum detection confidence (default 0.6).
        tracker:     Tracker config file (default bytetrack.yaml).

    Returns:
        Number of unique plates saved.
    """
    model = YOLO(model_path)

    if output_dir is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"cropped-detections/{video_name}"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Error reading video file: {video_path}"

    best_plates = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=conf,
            tracker=tracker,
            verbose=False,
        )
        result = results[0]

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            score = sharpness_score(crop)
            if track_id not in best_plates or score > best_plates[track_id]["score"]:
                best_plates[track_id] = {"score": score, "crop": crop.copy()}

    cap.release()

    for track_id, data in best_plates.items():
        save_path = os.path.join(output_dir, f"plate_track_{track_id}.jpg")
        cv2.imwrite(save_path, data["crop"])

    print(f"[Video] Saved {len(best_plates)} unique plates to '{output_dir}'")
    return len(best_plates)


def crop_plates_from_images(
    model_path: str,
    input_dir: str,
    output_dir: str,
    conf: float = 0.6,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> int:
    """
    Detects number plates in every image inside input_dir and saves
    all cropped detections to output_dir.

    Naming convention:  <original_stem>_det<N>.jpg
    where N is the zero-based index of each detection in that image.

    Args:
        model_path:  Path to the YOLO .pt weights file.
        input_dir:   Folder containing source images.
        output_dir:  Folder where cropped plates will be saved.
        conf:        Minimum detection confidence (default 0.6).
        extensions:  Tuple of accepted image file extensions.

    Returns:
        Total number of crops saved across all images.
    """
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ]

    if not image_files:
        print(f"[Images] No images found in '{input_dir}'")
        return 0

    total_saved = 0

    for filename in sorted(image_files):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [Skip] Could not read {filename}")
            continue

        results = model.predict(image, classes=[0], conf=conf, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        stem = os.path.splitext(filename)[0]

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            save_path = os.path.join(output_dir, f"{stem}_det{i}.jpg")
            cv2.imwrite(save_path, crop)
            total_saved += 1

        print(f"  {filename}: {len(boxes)} plate(s) detected")

    print(f"[Images] Saved {total_saved} crops to '{output_dir}'")
    return total_saved


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL = "/home/kaushik/Nepali_Num_plate/runs/detect/exp_imgsz640_ep80-2/weights/best.pt"

    # --- Video usage ---
    '''
    crop_plates_from_video(
        model_path=MODEL,
        video_path="/home/kaushik/Nepali_Num_plate/videos/video3.mp4",
    )
    '''
    # --- Image folder usage ---
    crop_plates_from_images(
        model_path=MODEL,
        input_dir="/home/kaushik/Nepali_Num_plate/images",
        output_dir="/home/kaushik/Nepali_Num_plate/cropped-plates",
    )