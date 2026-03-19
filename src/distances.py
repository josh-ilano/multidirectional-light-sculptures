import numpy as np
from scipy.ndimage import distance_transform_edt

# Helper functions to compute Euclidean distance fields of a binary silhouette to measure how far pixels lie 
# inside or outside the shape, enabling soft projection constraints, thickness estimation, 
# and more stable voxel carving / reconstruction.

def silhouette_distance_fields(mask: np.ndarray):
    """
    Function which computes the distance to background for foreground pixels and
    the distance to foreground for background pixels using Euclidean distance.
    """
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    return dist_inside, dist_outside

def outside_distance(mask: np.ndarray) -> np.ndarray:
    """
    Functino that returns 0 if inside mask, positive outside by distance to nearest foreground pixel.
    """
    return distance_transform_edt(~mask) * (~mask)

def inside_distance(mask: np.ndarray) -> np.ndarray:
    """
    Functino that returns positive if inside mask by distance to nearest background pixel.
    """
    return distance_transform_edt(mask) * mask