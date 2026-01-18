import os
import numpy as np
import cv2
import trimesh
import pyrender
import math

def normalize_mesh(mesh):
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    return mesh

def look_at_pyrender(cam_pos, target=np.array([0,0,0],dtype=float), up=np.array([0,1,0],dtype=float)):
    # pyrender camera looks along -Z
    forward = (target - cam_pos)
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)

    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=float)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward   # IMPORTANT for pyrender
    pose[:3, 3] = cam_pos
    return pose

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models3d")
obj_files = sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".obj")])
obj_path = os.path.join(MODELS_DIR, obj_files[0])
print("Using:", obj_path)

tm = trimesh.load(obj_path, force="mesh")
if isinstance(tm, trimesh.Scene):
    tm = trimesh.util.concatenate([g for g in tm.geometry.values()])
tm = normalize_mesh(tm)

mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
scene = pyrender.Scene(bg_color=[255,255,255,255])
scene.add(mesh)

camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
light = pyrender.DirectionalLight(color=[1,1,1], intensity=2.5)

renderer = pyrender.OffscreenRenderer(256, 256)

# one view only
radius = 3.0
cam_pos = np.array([radius, 0.2, 0.0], dtype=float)
pose = look_at_pyrender(cam_pos)

cam_node = scene.add(camera, pose=pose)
light_node = scene.add(light, pose=pose)

color, depth = renderer.render(scene)

scene.remove_node(cam_node)
scene.remove_node(light_node)
renderer.delete()

print("depth stats:", "min=", float(np.min(depth)), "max=", float(np.max(depth)))
print("finite depth pixels:", int(np.isfinite(depth).sum()))

# robust silhouette: finite AND > 0
mask = (np.isfinite(depth) & (depth > 0)).astype(np.uint8) * 255
print("mask white pixels:", int((mask > 0).sum()))

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "debug_views")
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, "debug_color.png"), color)
cv2.imwrite(os.path.join(out_dir, "debug_mask.png"), mask)
print("Saved debug_color.png and debug_mask.png to:", out_dir)
