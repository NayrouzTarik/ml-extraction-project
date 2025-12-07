from ultralytics import YOLO
import cv2
import os

def evaluate_yolo(model_path, val_folder):
    model = YOLO(model_path)
    results = model.val(data={"val": val_folder, "nc": 15})
    print("YOLO Evaluation Results:")
    print(results)
    return model

def detect_objects(model, img_path):
    results = model.predict(img_path)
    objects = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() 
        labels = result.boxes.cls.cpu().numpy()
        for box, label in zip(boxes, labels):
            objects.append({
                "bbox": box.tolist(),
                "label": int(label)
            })
    return objects

if __name__ == "__main__":
    val_folder = "datasets/val"  
    model = evaluate_yolo("backend/data/models/yolo15cats.pt", val_folder)
    print("Modèle prêt pour l'API")
