from flask import Flask, request, jsonify
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.pipeline import process_image
from backend.utils.similarity import compare_features, save_db, add_to_db, features_db as sim_features_db
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CATEGORIES = [
    "cat1", "cat2", "cat3", "cat4", "cat5",
    "cat6", "cat7", "cat8", "cat9", "cat10",
    "cat11", "cat12", "cat13", "cat14", "cat15"
]

if os.path.exists("backend/data/features_db.json"):
    with open("backend/data/features_db.json", "r") as f:
        features_db = json.load(f)
else:
    features_db = {}


# ENDPOINT: Recherche d'images similaires
@app.route("/search", methods=["POST"])
def search():
    """Rechercher les images similaires pour une image uploadée"""
    file = request.files["image"]
    path = "backend/data/uploads/" + file.filename
    os.makedirs("backend/data/uploads", exist_ok=True)
    file.save(path)
    
    query_objects = process_image(path)
    results = []

    for qobj in query_objects:
        best_match = None
        best_score = -1
        for img_path, objs in features_db.items():
            for obj in objs:
                score = compare_features(qobj, obj)
                if score > best_score:
                    best_score = score
                    best_match = img_path
        results.append({
            "query_label": qobj["label"],
            "category": CATEGORIES[qobj["label"]] if qobj["label"] < len(CATEGORIES) else "unknown",
            "best_match": best_match,
            "score": best_score
        })
    
    return jsonify(results)


# ENDPOINT: Afficher les descripteurs d'une image
@app.route("/descriptors", methods=["POST"])
def get_descriptors():
    """Extraire et afficher tous les descripteurs d'une image"""
    file = request.files["image"]
    path = "backend/data/uploads/" + file.filename
    os.makedirs("backend/data/uploads", exist_ok=True)
    file.save(path)
    
    # Extraire les descripteurs
    objects = process_image(path)
    
    # Formater la réponse avec les détails des descripteurs
    response = []
    for i, obj in enumerate(objects):
        response.append({
            "object_id": i,
            "label": obj["label"],
            "category": CATEGORIES[obj["label"]] if obj["label"] < len(CATEGORIES) else "unknown",
            "descriptors": {
                "hu_moments": {
                    "values": obj["hu"],
                    "description": "7 moments invariants de Hu (rotation, échelle, translation)"
                },
                "orientation_histogram": {
                    "values": obj["orientation_hist"],
                    "bins": 16,
                    "description": "Histogramme des orientations du gradient (16 bins)"
                },
                "hog": {
                    "length": len(obj["hog"]),
                    "description": "Histogram of Oriented Gradients (9 orientations, 16x16 pixels/cell)"
                },
                "lbp": {
                    "values": obj["lbp"],
                    "description": "Local Binary Pattern (P=8, R=1, uniform)"
                },
                "color_histogram": {
                    "length": len(obj["color_hist"]),
                    "channels": ["Blue", "Green", "Red"],
                    "bins_per_channel": 256,
                    "description": "Histogramme couleur BGR (256 bins par canal)"
                }
            }
        })
    
    return jsonify(response)


# ENDPOINT: Ajouter une image à la base de données
@app.route("/add", methods=["POST"])
def add_image():
    """Ajouter une image et ses descripteurs à la base de données"""
    file = request.files["image"]
    path = "backend/data/uploads/" + file.filename
    os.makedirs("backend/data/uploads", exist_ok=True)
    file.save(path)
    
    # Extraire et stocker les descripteurs
    objects = process_image(path)
    features_db[path] = objects
    
    # Sauvegarder la base
    with open("backend/data/features_db.json", "w") as f:
        json.dump(features_db, f, indent=2)
    
    return jsonify({
        "status": "success",
        "message": f"Image ajoutée avec {len(objects)} objets détectés",
        "path": path,
        "objects_count": len(objects)
    })


# ENDPOINT: Lister les catégories
@app.route("/categories", methods=["GET"])
def get_categories():
    """Retourner la liste des 15 catégories"""
    return jsonify({
        "count": len(CATEGORIES),
        "categories": [{"id": i, "name": cat} for i, cat in enumerate(CATEGORIES)]
    })


#ENDPOINT: Stats de la base de données
@app.route("/stats", methods=["GET"])
def get_stats():
    """Retourner les statistiques de la base de données"""
    total_objects = sum(len(objs) for objs in features_db.values())
    
    # Compter par catégorie
    category_counts = {cat: 0 for cat in CATEGORIES}
    for objs in features_db.values():
        for obj in objs:
            label = obj.get("label", -1)
            if 0 <= label < len(CATEGORIES):
                category_counts[CATEGORIES[label]] += 1
    
    return jsonify({
        "total_images": len(features_db),
        "total_objects": total_objects,
        "objects_per_category": category_counts
    })


# ENDPOINT: Health check
@app.route("/health", methods=["GET"])
def health():
    """Vérifier que l'API fonctionne"""
    return jsonify({"status": "ok", "message": "API is running"})


if __name__ == "__main__":
    print("API Flask démarrée sur http://localhost:5000")
    print("Endpoints disponibles:")
    print("   POST /search      - Rechercher images similaires")
    print("   POST /descriptors - Afficher les descripteurs d'une image")
    print("   POST /add         - Ajouter une image à la base")
    print("   GET  /categories  - Lister les 15 catégories")
    print("   GET  /stats       - Statistiques de la base")
    print("   GET  /health      - Health check")
    app.run(debug=True, port=5000)
