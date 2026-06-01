from pathlib import Path
import yaml

# dataset root path
dataset_dir = Path("/home/kaushik/Nepali_Num_plate/num_plate_dataset")

# YOLO dataset config
data = {
    "path": str(dataset_dir.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": ["number_plate"]
}

# write yaml file
yaml_path = dataset_dir / "data.yaml"

with open(yaml_path, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print(f"data.yaml created at: {yaml_path}")