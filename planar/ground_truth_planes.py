import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import h5py
import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
import trimesh
from plyfile import PlyData
from pytorch3d.renderer import RasterizationSettings, TexturesVertex
from pytorch3d.structures import Meshes
from tqdm.auto import tqdm

from common.scene_release import ScannetppScene_Release
from common.utils.utils import load_json, load_yaml_munch, read_txt_list, run_command
from planar.utils.encoding import decode_planar_colors, get_planar_colormap
from planar.utils.mesh import planar_segmentation
from planar.utils.renders import nerfstudio_to_colmap, process_frame


def process_scene_planar_mesh(scene: ScannetppScene_Release):
    filename = scene.scan_anno_json_path
    data = json.load(open(filename, "r"))
    aggregation = np.array(data["segGroups"])

    filename = scene.scan_sem_mesh_path
    plydata = PlyData.read(filename)
    vertices = plydata["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    faces = np.array(plydata["face"]["vertex_indices"])

    all_xyz = points.reshape(-1, 3)
    all_faces = faces.copy()

    group_segments = []
    group_labels = []
    for segmentIndex in range(len(aggregation)):
        group_segments.append(aggregation[segmentIndex]["segments"])
        group_labels.append(aggregation[segmentIndex]["label"])

    all_meshes = []
    all_labels = []
    all_planes = []

    offset = 0
    for group in group_segments:
        group = np.array(group).astype(np.int32)

        segment_indices = group.copy()
        xyz = all_xyz[segment_indices]

        # Filter faces
        faces = []
        segment_set = set(segment_indices)
        for face in all_faces:
            if all(v in segment_set for v in face):
                faces.append(face)
        faces = np.array(faces)

        # Renumber faces
        face_map = {old: new for new, old in enumerate(segment_indices)}
        faces = np.vectorize(face_map.get)(faces)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(xyz)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Apply segmentation
        mesh, labels, planes = planar_segmentation(mesh)
        all_meshes.append(mesh)

        # Remap labels
        unique_planar_labels = np.unique(labels[labels != -1])
        labels_map = {label: i + offset for i, label in enumerate(unique_planar_labels)}
        labels_map[-1] = -1  # keep background label

        remapped_labels = [labels_map[v] for v in labels]

        all_labels.extend(remapped_labels)
        all_planes.extend(planes)

        offset += len(planes)

    combined_labels = np.array(all_labels)
    combined_planes = np.array(all_planes)

    # Color mesh according to plane id labels
    num_labels = len(np.unique(combined_labels))
    num_planes = len(combined_planes)

    assert num_labels - 1 == num_planes

    colormap = get_planar_colormap(num_planes)

    # Save all colored meshes to a single file
    combined_mesh = o3d.geometry.TriangleMesh()
    for mesh in all_meshes:
        combined_mesh += mesh

    colors = colormap[combined_labels]
    colors = colors / 255.0
    combined_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    output_mesh_path = scene.planar_mesh_path
    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_mesh_path), combined_mesh)

    planar_params_path = scene.planar_params_path
    np.save(planar_params_path, combined_planes)


def process_scene_planar_mesh_renders(
    scene: ScannetppScene_Release, height: int, width: int
):
    planar_renders_dir = scene.planar_renders_dir
    planar_renders_dir.mkdir(exist_ok=True, parents=True)

    mesh_trimesh = trimesh.load_mesh(scene.planar_mesh_path)

    verts = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh_trimesh.faces, dtype=torch.int64)

    assert mesh_trimesh.visual is not None, "Mesh has no visual attributes"
    assert mesh_trimesh.visual.vertex_colors is not None, "Mesh has no vertex colors"
    assert len(mesh_trimesh.visual.vertex_colors) == len(verts), "Incorrect mesh colors"

    vertex_colors = torch.tensor(mesh_trimesh.visual.vertex_colors, dtype=torch.int64)

    plane_ids = decode_planar_colors(vertex_colors)

    alpha = torch.ones((vertex_colors.shape[0], 1), dtype=torch.float32)
    alpha[plane_ids == -1] = 0.0  # non-planar regions

    rgb = vertex_colors[:, :3] / 255.0
    vertex_colors_rgba = torch.cat([rgb, alpha], dim=1)

    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=TexturesVertex(verts_features=vertex_colors_rgba.unsqueeze(0)),
    )

    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    image_size = torch.tensor((height, width)).unsqueeze(0)

    transforms_filename = scene.iphone_nerfstudio_transform_undistorted_path

    with open(transforms_filename, "r") as f:
        transforms = json.load(f)

    assert transforms["camera_model"] == "PINHOLE"

    fx = transforms["fl_x"]
    fy = transforms["fl_y"]
    cx = transforms["cx"]
    cy = transforms["cy"]
    frame_h = transforms["h"]
    frame_w = transforms["w"]

    scale_x = width / frame_w
    scale_y = height / frame_h

    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y

    K = np.array(
        [
            [fx_scaled, 0, cx_scaled, 0],
            [0, fy_scaled, cy_scaled, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    K_b44 = torch.tensor(K, dtype=torch.float32).unsqueeze(0)

    for split in ["frames", "test_frames"]:

        for frame in transforms[split]:
            world_T_cam = np.array(frame["transform_matrix"])
            cam_T_world = nerfstudio_to_colmap(world_T_cam)
            cam_T_world_b44 = torch.tensor(cam_T_world, dtype=torch.float32).unsqueeze(
                0
            )

            frame_name = frame["file_path"]
            frame_name = Path(frame_name).stem
            frame_name = str(frame_name)

            process_frame(
                mesh,
                cam_T_world_b44,
                K_b44,
                image_size,
                raster_settings,
                frame_name,
                planar_renders_dir,
                render_depth=True,
            )

    run_command(
        f"tar -cf {planar_renders_dir}.tar -C {planar_renders_dir} .",
        verbose=True,
    )
    run_command(f"rm -rf {planar_renders_dir}", verbose=True)


def process_scene_hdf5(
    scene: ScannetppScene_Release, planar_height: int, planar_width: int
):
    transforms = load_json(scene.iphone_nerfstudio_transform_undistorted_path)

    height = int(transforms["h"])
    width = int(transforms["w"])

    camera_model = transforms["camera_model"]
    assert camera_model == "PINHOLE", f"Only PINHOLE camera model is supported"

    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )

    temp_images_dir = scene.iphone_data_dir / "undistorted_images"
    temp_renders_dir = scene.planar_renders_dir

    # Unpack the tar files for undistorted images and planar renders
    run_command(
        f"mkdir -p {temp_images_dir} && tar -xf {scene.iphone_data_dir}/rgb_undistorted.tar -C {temp_images_dir}",
        verbose=True,
    )
    run_command(
        f"mkdir -p {temp_renders_dir} && tar -xf {scene.planar_renders_dir}.tar -C {temp_renders_dir}",
        verbose=True,
    )

    hdf5_path = scene.planar_hdf5_path
    with h5py.File(hdf5_path, "w") as f:

        plane_params = np.load(scene.planar_params_path)  # (num_planes, 4)

        # same for all images
        f.create_dataset("intrinsics", data=K)
        f.create_dataset("plane_params", data=plane_params)

        for split in ["frames", "test_frames"]:
            frames = transforms[split]

            images = np.empty((len(frames), height, width, 3), dtype=np.uint8)
            depths = np.empty(
                (len(frames), planar_height, planar_width), dtype=np.float32
            )
            planes = np.empty(
                (len(frames), planar_height, planar_width), dtype=np.int16
            )
            poses = np.empty((len(frames), 4, 4), dtype=np.float32)

            for i, frame in enumerate(frames):

                frame_path = frame["file_path"]
                frame_name = Path(frame_path).stem

                # Load image
                img_path = temp_images_dir / frame_path
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[i] = img

                # Load depth
                depth_path = temp_renders_dir / f"{frame_name}_depth.npy"
                depth = np.load(depth_path)
                depths[i] = depth

                # Load plane ids
                plane_render_path = temp_renders_dir / f"{frame_name}_planes.png"

                plane_render = cv2.imread(str(plane_render_path))
                plane_render = cv2.cvtColor(plane_render, cv2.COLOR_BGR2RGB)
                plane_colors = plane_render[..., :3].astype(np.int64)
                plane_ids = decode_planar_colors(plane_colors).astype(np.int16)
                planes[i] = plane_ids

                # Load pose
                transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)
                poses[i] = transform_matrix

            prefix = ""
            if split == "test_frames":
                prefix = "test_"

            f.create_dataset(f"{prefix}frames", data=images)
            f.create_dataset(f"{prefix}depths", data=depths)
            f.create_dataset(f"{prefix}planes", data=planes)
            f.create_dataset(f"{prefix}poses", data=poses)

    # Clean up temporary directories
    run_command(f"rm -rf {temp_images_dir}", verbose=True)
    run_command(f"rm -rf {temp_renders_dir}", verbose=True)


def process_one_scene(scene_id, cfg):
    scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")

    process_scene_planar_mesh(scene)
    process_scene_planar_mesh_renders(scene, height=cfg.height, width=cfg.width)
    process_scene_hdf5(scene, planar_height=cfg.height, planar_width=cfg.width)


def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process, specify any one
    if cfg.get("scene_list_file"):
        scene_ids = read_txt_list(cfg.scene_list_file)
    elif cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    if cfg.get("scene_exclude_list_file"):
        exclude_scene_ids = read_txt_list(cfg.scene_exclude_list_file)
        scene_ids = [sid for sid in scene_ids if sid not in exclude_scene_ids]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_one_scene, sid, cfg): sid for sid in scene_ids
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="scene"):
            sid = futures[f]
            try:
                f.result()
            except BrokenProcessPool:
                print(
                    f"Scene {sid} failed with BrokenProcessPool (likely out of memory)"
                )
            except Exception as e:
                print(f"Scene {sid} failed with {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
