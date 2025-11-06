import numpy as np


def get_random_colormap(num_planes: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    colormap = rng.integers(0, 256, size=(num_planes + 1, 3), dtype=np.int64)
    colormap[-1] = 0
    return colormap


def get_planar_colormap(num_planes: int):
    ids = (np.arange(num_planes + 1, dtype=np.int64) + 1) * 100
    colormap = np.stack(
        [
            ids // (256 * 256),
            (ids // 256) % 256,
            ids % 256,
        ],
        axis=1,
    )
    colormap[-1] = 0
    return colormap


def decode_planar_colors(colors):
    plane_ids = (
        colors[..., 0].astype(np.int64) * 256 * 256
        + colors[..., 1].astype(np.int64) * 256
        + colors[..., 2].astype(np.int64)
    )
    plane_ids = plane_ids // 100 - 1  # shift to -1 as no plane
    return plane_ids
