import os

# backend/ folder
BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BACKEND_DIR, "data")

# Put PSB .off files here
PSB_OFF_DIR = os.path.join(DATA_DIR, "psb_off")

# Converted models (.obj) will be here
MODELS_DIR = os.path.join(DATA_DIR, "models3d")

PREVIEWS_DIR = os.path.join(DATA_DIR, "previews3d")
# Index file (features)
INDEX_PATH = os.path.join(DATA_DIR, "features_3d.json")

# Parameters
DEFAULT_N_VIEWS = 12
DEFAULT_IMG_SIZE = 256
