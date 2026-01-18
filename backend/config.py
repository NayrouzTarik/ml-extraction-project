"""
Configuration de l'application Flask.
"""
import os
from pathlib import Path

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Configuration Flask
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

# Dossiers pour les fichiers
UPLOAD_FOLDER = BASE_DIR / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Base de données
DATABASE_PATH = BASE_DIR / 'database.db'

# Modèle YOLO (fine-tuned)
YOLO_MODEL_PATH = PROJECT_ROOT / 'descriptors' / 'data+model' / 'best.pt'
DESCRIPTORS_MODULE_PATH = PROJECT_ROOT / 'descriptors'

# Dataset de validation
VAL_DESCRIPTORS_PATH = PROJECT_ROOT / 'descriptors' / 'data+model' / 'val_descriptors.pkl'
VAL_IMAGES_BASE_PATH = PROJECT_ROOT / 'descriptors' / 'data+model' / 'dataset' / 'val' / 'images'

# Classes YOLO (15 classes ImageNet custom - fine-tuned)
YOLO_CLASSES = [
    'baseball',             # 0
    'basketball',           # 1
    'orange',               # 2
    'lemon',                # 3
    'banana',               # 4
    'granny_smith',         # 5
    'corn',                 # 6
    'strawberry',           # 7
    'umbrella',             # 8
    'coffee_mug',           # 9
    'cucumber',             # 10
    'pineapple',            # 11
    'tiger_shark',          # 12
    'black_swan',           # 13
    'airliner'              # 14
]

# Créer les dossiers nécessaires
UPLOAD_FOLDER.mkdir(exist_ok=True)

