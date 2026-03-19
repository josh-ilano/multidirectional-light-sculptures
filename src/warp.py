import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

def smooth_displacement(dx: np.ndarray, dy: np.ndarray, sigma: float = 6.0):
    dx_s = gaussian_filter(dx, sigma=sigma)
    dy_s = gaussian_filter(dy, sigma=sigma)
    return dx_s, dy_s

def warp_mask(mask: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Backward warp a binary mask using displacement fields dx, dy.
    dx, dy are defined on image pixel grid.
    """
    h, w = mask.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    src_x = xx - dx
    src_y = yy - dy

    warped = map_coordinates(
        mask.astype(float),
        [src_y, src_x],
        order=1,
        mode="constant",
        cval=0.0
    )

    return warped >= 0.5