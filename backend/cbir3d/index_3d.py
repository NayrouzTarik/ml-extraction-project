import os
import json
import numpy as np

from cbir3d.render_views import render_views

from cbir3d.descriptor_hog import hog_from_views


# paths
CBIR3D_DIR = os.path.dirname(os.path.abspath(__file__))                 # website/backend/cbir3d
BACKEND_DIR = os.path.dirname(CBIR3D_DIR)                               # website/backend
DATA_DIR = os.path.join(BACKEND_DIR, "data")

MODELS_DIR = os.path.join(DATA_DIR, "models3d")                         # .obj here
INDEX_PATH = os.path.join(DATA_DIR, "features_3d.json")                 # output json

def build_index(limit: int | None = 50, n_views: int = 12, img_size: int = 256) -> dict:
    obj_files = sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".obj")])
    if not obj_files:
        raise FileNotFoundError(f"No .obj files in {MODELS_DIR}")

    if limit is not None:
        obj_files = obj_files[:limit]

    index = {}
    for i, fname in enumerate(obj_files, start=1):
        path = os.path.join(MODELS_DIR, fname)

        views = render_views(path, n_views=n_views, img_size=img_size)
        vec = hog_from_views(views)  # 7-dim

        index[fname] = {
            "vec": vec.tolist(),
            "meta": {"n_views": n_views, "img_size": img_size}
        }

        if i % 10 == 0:
            print(f"Indexed {i}/{len(obj_files)}")

    return index

def save_index(index: dict) -> None:
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)

if __name__ == "__main__":
    idx = build_index(limit=50, n_views=12, img_size=256)  # keep 50 for now
    save_index(idx)
    print("Saved index to:", INDEX_PATH)
    print("Total models indexed:", len(idx))
