import numpy as np
from scipy.ndimage import binary_erosion

def boundary_mask(mask: np.ndarray) -> np.ndarray:
    return mask & (~binary_erosion(mask))

def boundary_points(mask: np.ndarray):
    ys, xs = np.where(boundary_mask(mask))
    return np.stack([xs, ys], axis=1)

def nearest_boundary_point(mask: np.ndarray, target_xy):
    pts = boundary_points(mask)
    if len(pts) == 0:
        return None
    diffs = pts - np.asarray(target_xy)[None, :]
    d2 = np.sum(diffs * diffs, axis=1)
    return pts[np.argmin(d2)]