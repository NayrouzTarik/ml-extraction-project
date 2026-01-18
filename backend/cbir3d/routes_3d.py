import os
import json
import numpy as np
from flask import Blueprint, request, jsonify, send_from_directory 

from cbir3d.render_views import render_views
from cbir3d.descriptor_hog import hog_from_views as hu_from_views, cosine_distance

from cbir3d.preview_3d import ensure_preview_for_model, save_preview_for_uploaded_obj



api3d = Blueprint("api3d", __name__, url_prefix="/api3d")

CBIR3D_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CBIR3D_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "data")

MODELS_DIR = os.path.join(DATA_DIR, "models3d")
INDEX_PATH = os.path.join(DATA_DIR, "features_3d.json")
TMP_DIR = os.path.join(DATA_DIR, "tmp3d")


def load_index():
    if not os.path.exists(INDEX_PATH):
        return {}
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@api3d.route("/status", methods=["GET"])
def status():
    return jsonify({
        "models_dir": MODELS_DIR,
        "index_path": INDEX_PATH,
        "index_exists": os.path.exists(INDEX_PATH),
    })


@api3d.route("/models", methods=["GET"])
def list_models():
    # Optional query param: limit
    limit = int(request.args.get("limit", 80))

    obj_files = sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".obj")])[:limit]

    items = []
    for f in obj_files:
        preview_png = ensure_preview_for_model(f, n_views=12, img_size=256)
        items.append({
            "model": f,
            "preview_url": f"/previews3d/{preview_png}",
            "model_url": f"/api3d/model/{f}"
        })

    return jsonify({"status": "ok", "models": items})





@api3d.route("/search_by_name", methods=["POST"])
def search_by_name():
    index = load_index()
    if not index:
        return jsonify({"error": "Index not found or empty. Call /api3d/index first."}), 400

    body = request.get_json(silent=True) or {}
    model_name = body.get("model")
    top_k = int(body.get("top_k", 5))
    n_views = int(body.get("n_views", 12))
    img_size = int(body.get("img_size", 256))

    if not model_name:
        return jsonify({"error": "Missing model"}), 400

    query_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(query_path):
        return jsonify({"error": "Model not found in dataset"}), 404

    q_views = render_views(query_path, n_views=n_views, img_size=img_size)
    q_vec = hu_from_views(q_views)

    results = []
    for name, item in index.items():
        vec = np.array(item["vec"], dtype=np.float32)
        d = cosine_distance(q_vec, vec)
        preview_png = ensure_preview_for_model(name, n_views=n_views, img_size=img_size)

        results.append({"model": name, "distance": float(d), "preview_url": f"/previews3d/{preview_png}"})

    results.sort(key=lambda x: x["distance"])
    return jsonify({"status": "ok", "query": model_name, "top_k": results[:top_k]})


@api3d.route("/index", methods=["POST"])
def index_models():
    """
    Build index (like index_3d.py) via API.
    JSON body optional: {"limit": 50, "n_views": 12, "img_size": 256}
    """
    body = request.get_json(silent=True) or {}
    limit = body.get("limit", 50)
    n_views = int(body.get("n_views", 12))
    img_size = int(body.get("img_size", 256))

    obj_files = sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".obj")])
    if not obj_files:
        return jsonify({"error": f"No .obj files found in {MODELS_DIR}"}), 400

    if limit is not None:
        obj_files = obj_files[: int(limit)]

    index = {}
    for fname in obj_files:
        path = os.path.join(MODELS_DIR, fname)
        views = render_views(path, n_views=n_views, img_size=img_size)
        vec = hu_from_views(views)

        index[fname] = {"vec": vec.tolist(), "meta": {"n_views": n_views, "img_size": img_size}}

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)

    return jsonify({"status": "ok", "indexed": len(index), "index_path": INDEX_PATH})


@api3d.route("/search", methods=["POST"])
def search_models():
    """
    multipart/form-data:
      file: query.obj
      top_k: 5
      n_views: 12
      img_size: 256
    """
    index = load_index()
    if not index:
        return jsonify({"error": "Index not found or empty. Call /api3d/index first."}), 400

    if "file" not in request.files:
        return jsonify({"error": "Missing file field"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".obj"):
        return jsonify({"error": "Only .obj supported for query"}), 400

    top_k = int(request.form.get("top_k", 5))
    n_views = int(request.form.get("n_views", 12))
    img_size = int(request.form.get("img_size", 256))

    os.makedirs(TMP_DIR, exist_ok=True)
    query_path = os.path.join(TMP_DIR, f.filename)
    f.save(query_path)

    # compute query descriptor
    q_views = render_views(query_path, n_views=n_views, img_size=img_size)
    q_vec = hu_from_views(q_views)

    # create query preview (silhouette)
    query_preview = save_preview_for_uploaded_obj(
        obj_path=query_path,
        out_name=os.path.splitext(f.filename)[0] + "_query.png",
        n_views=n_views,
        img_size=img_size
    )

    # compute distances for each model in index
    results = []
    for name, item in index.items():
        vec = np.array(item["vec"], dtype=np.float32)
        d = cosine_distance(q_vec, vec)

        # ensure preview exists for this dataset model
        preview_png = ensure_preview_for_model(name, n_views=n_views, img_size=img_size)

        results.append({
            "model": name,
            "distance": float(d),
            "preview_url": f"/previews3d/{preview_png}"
        })

    results.sort(key=lambda x: x["distance"])

    return jsonify({
        "status": "ok",
        "query": f.filename,
        "query_preview_url": f"/previews3d/{query_preview}",
        "top_k": results[:top_k]
    })

@api3d.route("/model/<path:filename>", methods=["GET"])
def get_model_file(filename):
    # Serve dataset OBJ (models3d)
    return send_from_directory(MODELS_DIR, filename, as_attachment=False)
  
@api3d.route("/query/<path:filename>", methods=["GET"])
def get_query_file(filename):
    return send_from_directory(TMP_DIR, filename, as_attachment=False)
