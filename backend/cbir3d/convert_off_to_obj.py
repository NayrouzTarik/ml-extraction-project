import os
import trimesh

# Absolute paths (no package imports, Windows-safe)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # website/backend/cbir3d
BACKEND_DIR = os.path.dirname(THIS_DIR)                 # website/backend
DATA_DIR = os.path.join(BACKEND_DIR, "data")

PSB_ROOT = os.path.join(DATA_DIR, "psb_off")            # contains db/...
MODELS_DIR = os.path.join(DATA_DIR, "models3d")         # output folder


def find_off_files(root):
    off_files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".off"):
                off_files.append(os.path.join(dirpath, fn))
    off_files.sort()
    return off_files


def make_output_name(off_path):
    rel = os.path.relpath(off_path, PSB_ROOT)
    rel_no_ext = os.path.splitext(rel)[0]
    safe = rel_no_ext.replace(os.sep, "_").replace(" ", "_")
    return safe + ".obj"


def convert_all_off_to_obj(limit=50):
    os.makedirs(MODELS_DIR, exist_ok=True)

    off_files = find_off_files(PSB_ROOT)
    if not off_files:
        print("❌ No .off files found under:", PSB_ROOT)
        return 0

    off_files = off_files[:limit]

    count = 0
    for off_path in off_files:
        try:
            mesh = trimesh.load(off_path, force="mesh")
            out_name = make_output_name(off_path)
            out_path = os.path.join(MODELS_DIR, out_name)
            mesh.export(out_path)
            count += 1
            print(f"✅ {out_name}")
        except Exception as e:
            print(f"⚠️ Skip {off_path} → {e}")

    return count


if __name__ == "__main__":
    print("PSB_ROOT :", PSB_ROOT)
    print("MODELS_DIR:", MODELS_DIR)
    n = convert_all_off_to_obj(limit=50)
    print(f"\n🎯 Converted {n} OFF files to OBJ")
