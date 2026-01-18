import math
import numpy as np
import trimesh
import pyrender


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Center mesh at origin + scale to unit sphere.
    """
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)

    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    return mesh


def _look_at(camera_pos, target=np.array([0.0,0.0,0.0]), up=np.array([0.0,1.0,0.0])):
    camera_pos = camera_pos.astype(np.float64)
    forward = (target - camera_pos)
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)

    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward   # IMPORTANT: camera looks along -Z in pyrender
    pose[:3, 3] = camera_pos
    return pose


def render_views(obj_path: str, n_views: int = 12, img_size: int = 256) -> list[np.ndarray]:
    """
    Render N grayscale views of a 3D model around Y-axis.
    Returns: list of uint8 images (H x W)
    """
    tm = trimesh.load(obj_path, force="mesh")
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate([g for g in tm.geometry.values()])

    tm = normalize_mesh(tm)

    mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)

    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.6, 0.6, 0.6])
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)

    renderer = pyrender.OffscreenRenderer(img_size, img_size)

    views = []
    radius = 2.2
    height = 0.2

    for i in range(n_views):
        angle = 2 * math.pi * (i / n_views)

        cam_pos = np.array([
            radius * math.cos(angle),
            height,
            radius * math.sin(angle)
        ], dtype=np.float64)

        cam_pose = _look_at(cam_pos)

        cam_node = scene.add(camera, pose=cam_pose)
        light_node = scene.add(light, pose=cam_pose)

        color, depth = renderer.render(scene)

        scene.remove_node(cam_node)
        scene.remove_node(light_node)

        # RGB -> grayscale
        # depth is 0 where there's no object; >0 where object exists
        mask = (np.isfinite(depth) & (depth > 0)).astype(np.uint8) * 255

        mask = crop_and_resize_mask(mask, out_size=img_size, pad=8)
        views.append(mask)


    renderer.delete()
    return views


def crop_and_resize_mask(mask: np.ndarray, out_size: int = 256, pad: int = 8) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return mask.astype(np.uint8)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1]-1, x1 + pad); y1 = min(mask.shape[0]-1, y1 + pad)

    crop = mask[y0:y1+1, x0:x1+1]

    # keep aspect ratio using padding
    h, w = crop.shape
    s = max(h, w)
    canvas = np.zeros((s, s), dtype=np.uint8)
    yoff = (s - h) // 2
    xoff = (s - w) // 2
    canvas[yoff:yoff+h, xoff:xoff+w] = crop

    # resize back to out_size
    from PIL import Image
    canvas_img = Image.fromarray(canvas)
    canvas_img = canvas_img.resize((out_size, out_size), resample=Image.NEAREST)
    return np.array(canvas_img, dtype=np.uint8)
