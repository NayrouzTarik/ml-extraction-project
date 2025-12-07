from ultralytics import YOLO
import cv2

def load_yolo_model(path="backend/data/models/yolo15cats.pt"):
    return YOLO(path)

def detect_objects(model, image_path):
    results = model.predict(image_path)
    objects = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  
        labels = r.boxes.cls.cpu().numpy()  
        for i in range(len(boxes)):
            x1,y1,x2,y2 = boxes[i]
            label = int(labels[i])
            objects.append({"bbox":[x1,y1,x2,y2], "label":label})
    return objects
