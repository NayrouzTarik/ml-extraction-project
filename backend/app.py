"""
Application Flask principale pour l'API de recherche d'images.
Cette application sert d'interface entre le frontend Angular et les services de traitement d'images.
"""

from flask import Flask, send_from_directory
from flask_restful import Api
from flask_cors import CORS
from config import DEBUG, UPLOAD_FOLDER

# Importer tous les endpoints de l'API
# Chaque classe représente un endpoint REST qui gère une fonctionnalité spécifique
from api.resources import (
    UploadImage,          # Upload d'images avec détection d'objets
    GetDescriptors,       # Récupération des descripteurs d'une image
    SearchSimilar,        # Recherche d'images similaires
    GetImages,            # Liste de toutes les images uploadées
    DeleteImage,          # Suppression d'une image
    GetObject,            # Détails d'un objet spécifique
    TransformImage        # Transformation d'images (crop, resize, rotate)
)

# Créer l'application Flask principale
app = Flask(__name__)
# 3D CBIR routes (view-based retrieval)
from cbir3d.routes_3d import api3d
app.register_blueprint(api3d)

app.config['DEBUG'] = DEBUG  # Mode debug activé ou non selon la configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16 MB pour les fichiers uploadés

# Activer CORS (Cross-Origin Resource Sharing)
# Permet au frontend Angular (sur localhost:4200) de communiquer avec le backend (sur localhost:5000)
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/api3d/*": {"origins": "*"}
})


# Créer l'API Flask-RESTful
# Tous les endpoints seront préfixés par /api
api = Api(app, prefix='/api')

# Enregistrer chaque endpoint avec sa route correspondante
api.add_resource(UploadImage, '/upload')                              # POST /api/upload
api.add_resource(GetDescriptors, '/descriptors/<int:image_id>')       # GET /api/descriptors/1
api.add_resource(SearchSimilar, '/search')                            # POST /api/search
api.add_resource(GetImages, '/images')                                # GET /api/images
api.add_resource(DeleteImage, '/images/<int:image_id>')               # DELETE /api/images/1
api.add_resource(GetObject, '/objects/<int:object_id>')               # GET /api/objects/1
api.add_resource(TransformImage, '/transform')                        # POST /api/transform



@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """
    Endpoint pour servir les fichiers images uploadés.
    Permet au frontend d'afficher les images via une URL HTTP.
    Exemple: /uploads/20251219_135125_fraise.jpg
    """
    return send_from_directory(str(UPLOAD_FOLDER), filename)

@app.route('/previews3d/<path:filename>')
def previews3d_file(filename):
    return send_from_directory('data/previews3d', filename)


@app.route('/val_images/<path:filename>')
def val_image(filename):
    """
    Endpoint pour servir les images du dataset de validation.
    Ces images sont utilisées pour la recherche de similaires.
    """
    from config import VAL_IMAGES_BASE_PATH
    if VAL_IMAGES_BASE_PATH.exists():
        return send_from_directory(str(VAL_IMAGES_BASE_PATH), filename)
    return {'error': 'Dataset de validation non disponible'}, 404


@app.route('/api/health')
def health_check():
    """
    Endpoint de santé pour vérifier que l'API fonctionne.
    Utile pour les tests et la vérification du statut du serveur.
    """
    return {'status': 'ok', 'message': 'API is running'}, 200


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Démarrage de l'API Flask")
    print("=" * 60)
    print(f"📡 Endpoints disponibles:")
    print(f"   POST   /api/upload              - Upload d'images")
    print(f"   GET    /api/descriptors/<id>    - Récupérer descripteurs")
    print(f"   POST   /api/search              - Recherche similaire")
    print(f"   GET    /api/images              - Liste des images")
    print(f"   DELETE /api/images/<id>         - Supprimer image")
    print(f"   GET    /api/objects/<id>        - Récupérer objet")
    print(f"   GET    /api/health              - Health check")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=DEBUG)

