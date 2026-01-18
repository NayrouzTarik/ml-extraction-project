import os
from render_views import render_views
from descriptor_hu import hu_from_views

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models3d")
obj_files = sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".obj")])

obj_path = os.path.join(MODELS_DIR, obj_files[0])
print("Using:", obj_path)

views = render_views(obj_path, n_views=12, img_size=256)
vec = hu_from_views(views)

print("Descriptor dimension:", vec.shape[0])
print("Values:", vec)
