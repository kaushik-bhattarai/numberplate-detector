from pathlib import Path

from ultralytics import YOLO

'''
model = YOLO("yolo26m.pt")

model.train(
    data="/home/kaushik/Nepali_Num_plate/num_plate_dataset/data.yaml",
    epochs=80,       
    imgsz=640,
    batch=8,        
    name="exp_imgsz640_ep80"
)'''

#resuming training 
model = YOLO("runs/detect/exp_imgsz640_ep80-2/weights/last.pt")

model.train(resume=True)