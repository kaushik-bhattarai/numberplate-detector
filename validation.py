from ultralytics import YOLO, solutions
model = YOLO("/home/kaushik/Nepali_Num_plate/runs/detect/exp_imgsz640_ep80-2/weights/best.pt")

#metrics = model.val(data="/home/kaushik/Nepali_Num_plate/num_plate_dataset/data.yaml")

#print(metrics)

#test on a single image
results = model.predict(
    source="/home/kaushik/Nepali_Num_plate/num_plate_dataset/images/train/2024-03-08_23_10.jpg",
    conf=0.25,
    save=True
)

for r in results:
    print(r.boxes.xyxy)
    print(r.boxes.conf)
