import numpy as np
from projections import project_points_orthographic

def compute_shadow_hull(sources, voxel_centers):
    """
    Intersect all shadow cones in voxelized form.
    A voxel is kept only if its projection lies inside every silhouette.
    """
    nx, ny, nz, _ = voxel_centers.shape
    pts = voxel_centers.reshape(-1, 3)

    keep = np.ones(len(pts), dtype=bool)

    for src in sources:
        px, py, valid, _ = project_points_orthographic(
            pts,
            src.direction,
            src.up,
            src.world_center,
            src.world_size,
            src.image.shape
        )

        pxi = np.clip(np.round(px).astype(int), 0, src.image.shape[1] - 1)
        pyi = np.clip(np.round(py).astype(int), 0, src.image.shape[0] - 1)

        inside = np.zeros(len(pts), dtype=bool)
        inside[valid] = src.image[pyi[valid], pxi[valid]]

        keep &= inside

    return keep.reshape(nx, ny, nz)