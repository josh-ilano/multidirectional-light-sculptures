from dataclasses import dataclass
import numpy as np

@dataclass
class ShadowSource:
    image: np.ndarray          # bool array [H, W], True = shadow/filled pixel
    direction: np.ndarray      # (3,) unit vector, orthographic light/view direction
    up: np.ndarray             # (3,) unit vector defining image vertical axis
    world_center: np.ndarray   # (3,) center of volume
    world_size: float          # cube side length in world units