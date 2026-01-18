import os
import json
import numpy as np

from render_views import render_views
from descriptor_hu import hu_from_views, cosine_distance

CBIR3D_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CBIR3D_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "data")

MODELS_DIR = os.path.join(DATA_DIR, "models3d")
INDEX_PATH = os.path.join(DATA_DIR, "features_3d.json")

def load_index():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    index = load_index()

    # pick one obj as query
    obj_files = sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".obj")])
    query_name = obj_files[0]
    query_path = os.path.join(MODELS_DIR, query_name)

    q_views = render_views(query_path, n_views=12, img_size=256)
    q_vec = hu_from_views(q_views)

    results = []
    for name, item in index.items():
        vec = np.array(item["vec"], dtype=np.float32)
        d = cosine_distance(q_vec, vec)
        results.append((name, d))

    results.sort(key=lambda x: x[1])

    print("Query:", query_name)
    print("Top 5:")
    for name, d in results[:5]:
        print(name, "dist=", d)
