import os
import numpy as np
from PIL import Image
from render import render_shadow

def save_grayscale(arr_uint8, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr_uint8).save(path)

def save_bool_mask(mask, path):
    img = (mask.astype(np.uint8) * 255)
    save_grayscale(img, path)

def make_comparison_image(target, actual):
    """
    RGB comparison:
    - white = background
    - black = correct shadow overlap
    - red = missing target pixels
    - blue = extra shadow pixels
    """
    h, w = target.shape
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 255

    overlap = target & actual
    missing = target & (~actual)
    extra = (~target) & actual

    rgb[overlap] = [0, 0, 0]
    rgb[missing] = [255, 0, 0]
    rgb[extra] = [0, 0, 255]

    return rgb

def evaluate_view(target, actual):
    inter = np.logical_and(target, actual).sum()
    union = np.logical_or(target, actual).sum()
    iou = inter / max(union, 1)

    missing = target & (~actual)
    extra = (~target) & actual

    return {
        "iou": float(iou),
        "missing_pixels": int(missing.sum()),
        "extra_pixels": int(extra.sum()),
        "target_pixels": int(target.sum()),
        "actual_pixels": int(actual.sum()),
    }

def simulate_and_save(voxels, voxel_centers, sources, out_dir, prefix):
    """
    Render all views and save:
    - target
    - actual
    - missing
    - extra
    - rgb comparison
    """
    os.makedirs(out_dir, exist_ok=True)
    summaries = []

    for i, src in enumerate(sources):
        target = src.image
        actual = render_shadow(voxels, voxel_centers, src)

        missing = target & (~actual)
        extra = (~target) & actual
        comp = make_comparison_image(target, actual)
        metrics = evaluate_view(target, actual)

        save_bool_mask(target, os.path.join(out_dir, f"{prefix}_view{i}_target.png"))
        save_bool_mask(actual, os.path.join(out_dir, f"{prefix}_view{i}_actual.png"))
        save_bool_mask(missing, os.path.join(out_dir, f"{prefix}_view{i}_missing.png"))
        save_bool_mask(extra, os.path.join(out_dir, f"{prefix}_view{i}_extra.png"))
        save_grayscale(comp, os.path.join(out_dir, f"{prefix}_view{i}_comparison.png"))

        summaries.append(metrics)

    return summaries