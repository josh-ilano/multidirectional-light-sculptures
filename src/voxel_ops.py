import numpy as np

def make_voxel_centers(nx, ny, nz, world_size):
    xs = np.linspace(-0.5, 0.5, nx, endpoint=False) + 0.5 / nx
    ys = np.linspace(-0.5, 0.5, ny, endpoint=False) + 0.5 / ny
    zs = np.linspace(-0.5, 0.5, nz, endpoint=False) + 0.5 / nz

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1) * world_size
    return pts  # [nx, ny, nz, 3]

def voxel_pitch(world_size, nx, ny, nz):
    return world_size / max(nx, ny, nz)