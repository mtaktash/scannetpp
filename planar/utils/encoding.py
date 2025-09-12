import numpy as np


def get_planar_colormap(num_planes: int):
    segmentation_color = (np.arange(num_planes + 1) + 1) * 100
    segmentation_color = segmentation_color.astype(np.int64)
    colormap = np.stack(
        [
            segmentation_color // (256 * 256),
            (segmentation_color // 256) % 256,
            segmentation_color % 256,
        ],
        axis=1,
    )
    colormap[-1] = 0
    return colormap


def decode_planar_colors(colors):
    plane_ids = colors[..., 0] * 256 * 256 + colors[..., 1] * 256 + colors[..., 2]
    plane_ids = plane_ids // 100 - 1
    return plane_ids
