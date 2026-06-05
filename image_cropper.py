import os
import cv2
from ultralytics import YOLO

# Load model
model = YOLO(
    "/home/kaushik/Nepali_Num_plate/runs/detect/exp_imgsz640_ep80-2/weights/best.pt"
)

#videopath
video_path = "/home/kaushik/Nepali_Num_plate/videos/video3.mp4"

# Create output folder based on video name
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_dir = f"cropped-detections/{video_name}"
os.makedirs(output_dir, exist_ok=True)

# Video
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Store best crop for each tracked plate
best_plates = {}

def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.6,
        tracker="bytetrack.yaml",
        verbose=False
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

        if (
            track_id not in best_plates
            or score > best_plates[track_id]["score"]
        ):
            best_plates[track_id] = {
                "score": score,
                "crop": crop.copy()
            }

cap.release()

# Save one image per tracked plate
for track_id, data in best_plates.items():

    save_path = os.path.join(output_dir, f"plate_track_{track_id}.jpg")
    cv2.imwrite(save_path, data["crop"]
    )

print(f"Saved {len(best_plates)} unique plates.")