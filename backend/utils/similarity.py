import json
import numpy as np
from scipy.spatial.distance import cosine
from backend.utils.pipeline import process_image

features_db = {}  

def add_to_db(image_path):
    features_db[image_path] = process_image(image_path)

def save_db(path="backend/data/features_db.json"):
    with open(path,"w") as f:
        json.dump(features_db,f,indent=2)

def compare_features(obj1,obj2):
    scores = []
    for key in ["hu","orientation_hist","hog","lbp","color_hist"]:
        f1 = np.array(obj1[key])
        f2 = np.array(obj2[key])
        score = 1 - cosine(f1,f2)
        scores.append(score)
    return np.mean(scores)
