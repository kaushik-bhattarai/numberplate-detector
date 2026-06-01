#structuring the dataset into YOLO format

'''
num_plate_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
    ├── train/
    │   ├── img_001.txt
    │   └── ...
    └── val/
       └── ...
'''
import shutil
from pathlib import Path

dataset_dir = Path("/home/kaushik/Nepali_Num_plate/num_plate_dataset")

images_dir = dataset_dir / "images"
labels_src = dataset_dir / "labels"

train_img_dir = images_dir / "train"
val_img_dir = images_dir / "val"
train_lbl_dir = dataset_dir / "labels" / "train"
val_lbl_dir = dataset_dir / "labels" / "val"

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    d.mkdir(parents=True, exist_ok=True)

images = list(images_dir.glob("*.jpg"))

split = int(0.8 * len(images))
train_imgs = images[:split]
val_imgs = images[split:]

def move_files(img_list, img_dst, lbl_dst):
    for img in img_list:
        label = labels_src / (img.stem + ".txt")

        shutil.move(str(img), str(img_dst / img.name))

        if label.exists():
            shutil.move(str(label), str(lbl_dst / label.name))

move_files(train_imgs, train_img_dir, train_lbl_dir)
move_files(val_imgs, val_img_dir, val_lbl_dir)

print("Dataset  move completed.")