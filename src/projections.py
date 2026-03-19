import numpy as np

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector")
    return v / n

def make_camera_basis(direction, up):
    w = normalize(direction)
    u = normalize(np.cross(up, w))
    v = normalize(np.cross(w, u))
    return u, v, w

def project_points_orthographic(points, direction, up, world_center, world_size, image_shape):
    """
    points: [N, 3]
    returns:
      px, py: projected pixel coords (float)
      valid: bool mask
      depth: projection onto view direction
    """
    H, W = image_shape
    u, v, w = make_camera_basis(direction, up)

    rel = points - world_center[None, :]
    x = rel @ u
    y = rel @ v
    z = rel @ w

    half = world_size / 2.0

    px = ((x + half) / world_size) * (W - 1)
    py = ((half - y) / world_size) * (H - 1)

    valid = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)

    return px, py, valid, z