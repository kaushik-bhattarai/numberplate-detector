from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

metrics = model.val(data="/home/kaushik/Nepali_Num_plate/num_plate_dataset/data.yaml")

print(metrics)