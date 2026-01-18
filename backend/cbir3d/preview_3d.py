import os
import cv2
import numpy as np
from PIL import Image
from cbir3d.render_views import render_views
from cbir3d.config_3d import PREVIEWS_DIR, MODELS_DIR



CBIR3D_DIR = os.path.dirname(os.path.abspath(__file__))   # website/backend/cbir3d
BACKEND_DIR = os.path.dirname(CBIR3D_DIR)                 # website/backend
DATA_DIR = os.path.join(BACKEND_DIR, "data")

MODELS_DIR = os.path.join(DATA_DIR, "models3d")
PREVIEWS_DIR = os.path.join(DATA_DIR, "previews3d")


def ensure_preview_for_model(model_name: str, n_views=12, img_size=256) -> str:
    """
    Generates preview PNG for a dataset model using the SAME render_views() as descriptors.
    Always saves view0 (first view) so what you see == what was compared.
    """
    os.makedirs(PREVIEWS_DIR, exist_ok=True)

    base = os.path.splitext(model_name)[0]
    out_name = f"{base}.png"
    out_path = os.path.join(PREVIEWS_DIR, out_name)

    # Always regenerate to avoid stale cache while debugging
    model_path = os.path.join(MODELS_DIR, model_name)

    views = render_views(model_path, n_views=n_views, img_size=img_size)
    img = views[0]  # ✅ view0 exactly like descriptor pipeline

    Image.fromarray(img).save(out_path)
    return out_name


def ensure_preview_for_query(obj_path: str, out_base_name: str, n_views=12, img_size=256) -> str:
    os.makedirs(PREVIEWS_DIR, exist_ok=True)

    out_name = f"{out_base_name}_query.png"
    out_path = os.path.join(PREVIEWS_DIR, out_name)

    views = render_views(obj_path, n_views=n_views, img_size=img_size)
    img = views[0]  # ✅ view0

    Image.fromarray(img).save(out_path)
    return out_name

def save_preview_for_uploaded_obj(obj_path: str, out_name: str, n_views: int = 12, img_size: int = 256) -> str:
    """
    Creates a preview for an uploaded query obj (path given), stored in PREVIEWS_DIR with provided name.
    Returns preview filename (png).
    """
    os.makedirs(PREVIEWS_DIR, exist_ok=True)

    preview_name = out_name if out_name.lower().endswith(".png") else (out_name + ".png")
    preview_path = os.path.join(PREVIEWS_DIR, preview_name)

    views = render_views(obj_path, n_views=n_views, img_size=img_size)
    img = views[0]
    cv2.imwrite(preview_path, img)
    return preview_name
