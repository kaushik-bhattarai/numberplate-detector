
# copies all the images that were labeled in lable studio into /verified-plates folder.
# /corrected-plates folder was then deleted.


import json
import os
import shutil

EXPORT_PATH = "json_files/final_data.json"
IMAGE_DIR = "corrected-plates"
DEST_DIR = "verified-plates"    

os.makedirs(DEST_DIR, exist_ok=True)

# Load exported tasks
with open(EXPORT_PATH, encoding="utf-8") as f:
    data = json.load(f)

# Get filenames of images still in Label Studio
kept_images = set()
for task in data:
    img_path = task["data"]["image"]
    img_name = os.path.basename(img_path.split("?d=")[-1])
    kept_images.add(img_name)

print(f"Images in Label Studio: {len(kept_images)}")

# Move kept images to verified-plates/
moved = 0
missing = 0
for img_name in sorted(kept_images):
    src = os.path.join(IMAGE_DIR, img_name)
    dst = os.path.join(DEST_DIR, img_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)   # copy2 preserves metadata
        moved += 1
    else:
        print(f"Missing: {img_name}")
        missing += 1

print(f"\nCopied {moved} images to '{DEST_DIR}/'")
print(f"Missing files: {missing}")
print(f"Original '{IMAGE_DIR}/' folder untouched")