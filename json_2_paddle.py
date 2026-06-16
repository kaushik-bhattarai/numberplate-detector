import json
import os
import shutil
from PIL import Image

EXPORT_PATH = "json_files/final_data_characters-less.json"
IMAGE_DIR = "verified-plates"
OUTPUT_DIR = "paddle_dataset_V3" #V2-> corrected version,  v3-> removed characters
TRAIN_LABEL = os.path.join(OUTPUT_DIR, "train_label_rec.txt")
VAL_LABEL = os.path.join(OUTPUT_DIR, "val_label_rec.txt")

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

with open(EXPORT_PATH, encoding="utf-8") as f:
    data = json.load(f)

def get_text(task):
    boxes = {}
    texts = {}
    for r in task['annotations'][0]['result']:
        rid = r['id']
        if r['type'] == 'rectanglelabels':
            boxes[rid] = (r['value']['y'], r['value']['x'])
        elif r['type'] == 'textarea':
            texts[rid] = r['value']['text'][0].strip()
    # sort by row (y) then column (x) for same-row boxes
    paired = [(boxes[rid], texts[rid]) for rid in boxes if rid in texts]
    paired.sort(key=lambda x: (round(x[0][0] / 10), x[0][1]))
    return "".join([t.replace(" ", "") for _, t in paired if t])
records = []

for task in data:
    img_path_raw = task["data"]["image"]
    img_name = os.path.basename(img_path_raw.split("?d=")[-1])
    img_src = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_src):
        print(f"Missing: {img_name}")
        continue

    annotations = task.get("annotations", [])
    if not annotations or not annotations[0].get("result"):
        print(f"No annotations: {img_name}")
        continue

    text = get_text(task)
    if not text:
        continue

    shutil.copy2(img_src, os.path.join(OUTPUT_DIR, "images", img_name))
    records.append((f"images/{img_name}", text))

# Split 80/20
split = int(len(records) * 0.8)
train_records = records[:split]
val_records = records[split:]

with open(TRAIN_LABEL, "w", encoding="utf-8") as f:
    for img_path, text in train_records:
        f.write(f"{img_path}\t{text}\n")

with open(VAL_LABEL, "w", encoding="utf-8") as f:
    for img_path, text in val_records:
        f.write(f"{img_path}\t{text}\n")

print(f"Train: {len(train_records)} | Val: {len(val_records)}")

# Preview first 5
print("\nSample labels:")
for img, text in records[:5]:
    print(f"  {img}  →  {text}")