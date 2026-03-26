from dataclasses import dataclass
import numpy as np

@dataclass
class ShadowSource:
    image: np.ndarray          # boolean 2D silhouette matrix where True = filled pixel
    direction: np.ndarray      # unit vector, orthographic light/view direction
    up: np.ndarray             # unit vector defining image vertical axis
    world_center: np.ndarray   # center of volume
    world_size: float          # cube side length in world units

# Takes as input the images and world size and assign a default orthographic direction/up pair
def build_sources(images, world_size):
    # default view configs: X, Y, Z
    view_configs = [
        {
            "direction": np.array([1, 0, 0], dtype=float),
            "up": np.array([0, 1, 0], dtype=float),
        },
        {
            "direction": np.array([0, 0, 1], dtype=float),
            "up": np.array([0, 1, 0], dtype=float),
        },
        {
            "direction": np.array([0, 1, 0], dtype=float),
            "up": np.array([0, 0, 1], dtype=float),
        },
    ]

    # for each image, build the shadow source
    sources = []
    for i, img in enumerate(images):
        cfg = view_configs[i]
        sources.append(
            ShadowSource(
                image=img,
                direction=cfg["direction"],
                up=cfg["up"],
                world_center=np.array([0, 0, 0], dtype=float),
                world_size=world_size,
            )
        )

    return sources