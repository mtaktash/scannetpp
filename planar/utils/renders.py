from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


def interpolate_face_attributes_nearest(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    """Given some attributes for vertices of a face and the coordinates of a point,
    this function interpolates by nearest interpolation the attributes.

    Inspired by:
        https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/interp_face_attrs.html

    Licensed under BSD License:

    For PyTorch3D software

    Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    * Neither the name Meta nor the names of its contributors may be used to
    endorse or promote products derived from this software without specific
    prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Params:
        pix_to_face:
        barycentric_coords: coordinates of the point
        face_attributes:
    """
    F, FV, D = face_attributes.shape
    N, H, W, K, _ = barycentric_coords.shape

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = pix_to_face < 0
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = face_attributes.gather(0, idx).view(N, H, W, K, 3, D)
    barycentric_coords = (
        barycentric_coords.amax(-1, True) == barycentric_coords
    ).float()
    pixel_vals = (barycentric_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0  # Replace masked values in output.
    return pixel_vals


def nerfstudio_to_colmap(c2w: np.ndarray):
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1
    # world coordinate transform: inverse for map colmap gravity guess (-y) to nerfstudio convention (+z)
    # to use SfM pointcloud
    # https://github.com/nerfstudio-project/nerfstudio/blob/ec10c49d51cfebc52618ece1221ec4511ac19b67/nerfstudio/data/dataparsers/colmap_dataparser.py#L169
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[2, :] *= -1
    w2c = np.linalg.inv(c2w)
    return w2c


@torch.no_grad()
def process_frame(
    mesh: Meshes,
    cam_T_world_b44: torch.Tensor,
    K_b44: torch.Tensor,
    image_size: torch.Tensor,
    raster_settings: RasterizationSettings,
    frame_name: str,
    save_path: Path,
    render_depth: bool,
):
    """Render a frame.
    Params:
        mesh: mesh to render, as pytorch3d.structures.Meshes
        cam_T_world_b44: camera pose from world to camera, as (b,4,4) tensor
        K_b44: camera intrinsics, as (b,4,4) tensor
        image_size: size of the image to render, as a tensor with two elements (1,H,W)
        raster_settings: setting for the rasterizer
        frame_name: final name of the frame
        save_path: path to the folder in which we are going to save the frame
        render_depth: whether to render depth information
    """

    R = cam_T_world_b44[:, :3, :3]
    T = cam_T_world_b44[:, :3, 3]
    K = K_b44[:, :3, :3]

    cams = cameras_from_opencv_projection(
        R=R, tvec=T, camera_matrix=K, image_size=image_size
    )

    rasterizer = MeshRasterizer(
        cameras=cams,
        raster_settings=raster_settings,
    )

    _mesh = mesh.extend(len(cams))
    fragments = rasterizer(_mesh)

    # nearest sampling
    faces_packed = _mesh.faces_packed()
    assert _mesh.textures is not None, "Mesh has no textures"

    verts_features_packed = _mesh.textures.verts_features_packed()
    faces_verts_features = verts_features_packed[faces_packed]
    texture_bhw14 = interpolate_face_attributes_nearest(
        fragments.pix_to_face, fragments.bary_coords, faces_verts_features
    )
    rendered_depth_bhw = fragments.zbuf[..., 0]

    plane_ids = texture_bhw14.cpu().numpy()[0, ..., 0, :]
    rendered_depth = rendered_depth_bhw.cpu().numpy().squeeze()

    # save image with plane ids
    plane_ids = (plane_ids * 255).astype(np.uint8)
    plane_ids = Image.fromarray(plane_ids)
    plane_ids.save(save_path / f"{frame_name}_planes.png")

    if render_depth:
        np.save(save_path / f"{frame_name}_depth.npy", rendered_depth)
