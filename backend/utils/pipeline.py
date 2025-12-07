from backend.utils.yolo_detector import load_yolo_model, detect_objects
from backend.utils.features.hu_moments import extract_hu_moments, visualize_hu_moments
from backend.utils.features.orientation import extract_orientation_histogram, visualize_orientation
from backend.utils.features.hog_descriptor import extract_hog, visualize_hog
from backend.utils.features.lbp_descriptor import extract_lbp, visualize_lbp
from backend.utils.features.color_histogram import extract_color_hist, visualize_color_hist
import cv2
import numpy as np

yolo_model = load_yolo_model()

def process_image(image_path):
    objects = detect_objects(yolo_model, image_path)
    img = cv2.imread(image_path)
    if img is None:
        return []
        
    results = []
    for obj in objects:
        x1,y1,x2,y2 = map(int,obj["bbox"])
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        descriptors = {
            "bbox": obj["bbox"],
            "label": obj["label"],
            "hu": extract_hu_moments(crop).tolist(),
            "orientation_hist": extract_orientation_histogram(crop).tolist(),
            "hog": extract_hog(crop).tolist(),
            "lbp": extract_lbp(crop).tolist(),
            "color_hist": extract_color_hist(crop).tolist(),
            
            "visualizations": {
                "hu": visualize_hu_moments(crop),
                "orientation": visualize_orientation(crop),
                "hog": visualize_hog(crop),
                "lbp": visualize_lbp(crop),
                "color_hist": visualize_color_hist(crop)
            }
        }
        results.append(descriptors)
    return results
