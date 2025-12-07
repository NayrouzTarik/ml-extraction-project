from ultralytics import YOLO
import sys

def main():
    #Aya's path
    model_path = "backend/data/models/yolo15cats.pt" 
    try:
        model = YOLO(model_path)
    except Exception:
        print(f"Model not found at {model_path}, utilizing 'yolov8n.pt' for demonstration.")
        model = YOLO("yolov8n.pt")

    print(f"🚀 Evaluating model: {model.ckpt_path if hasattr(model, 'ckpt_path') else model_path}")

    data_config = "datasets/data.yaml" 
    
    try:
        metrics = model.val(data=data_config)
        
        print("\nEvaluation Complete!")
        print(f"mAP@50: {metrics.box.map50}")
        print(f"mAP@50-95: {metrics.box.map}")
        
    except Exception as e:
        print(f"\n Evaluation Failed: {e}")
        print("Tip: Ensure 'datasets/data.yaml' exists and points to valid images.")

if __name__ == "__main__":
    main()
