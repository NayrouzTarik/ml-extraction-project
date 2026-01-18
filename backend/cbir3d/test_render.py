import os
import cv2
from render_views import render_views

# pick one converted model
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models3d")
obj_files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".obj")]
obj_files.sort()

obj_path = os.path.join(MODELS_DIR, obj_files[0])
print("Using:", obj_path)

views = render_views(obj_path, n_views=12, img_size=256)

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "debug_views")
os.makedirs(out_dir, exist_ok=True)

for i, img in enumerate(views):
    cv2.imwrite(os.path.join(out_dir, f"view_{i:02d}.png"), img)

print("Saved views to:", out_dir)
