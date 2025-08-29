import argparse
import collections
import json
import sys
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm

try:
    import renderpy
except ImportError:
    print(
        "renderpy not installed. Please install renderpy from https://github.com/liu115/renderpy"
    )
    sys.exit(1)


from common.scene_release import ScannetppScene_Release
from common.utils.colmap import Camera, read_model
from common.utils.utils import load_yaml_munch, read_txt_list

Image = collections.namedtuple("Image", ["id", "world_to_camera", "camera_id", "name"])


def read_nerfstudio_model(path):
    with open(path, "r") as f:
        data = json.load(f)

    width = data["w"]
    height = data["h"]

    fx = data["fl_x"]
    fy = data["fl_y"]
    cx = data["cx"]
    cy = data["cy"]

    camera_model = data["camera_model"]

    k1 = data.get("k1", 0.0)
    k2 = data.get("k2", 0.0)
    k3 = data.get("k3", 0.0)
    k4 = data.get("k4", 0.0)
    p1 = data.get("p1", 0.0)
    p2 = data.get("p2", 0.0)

    params = [fx, fy, cx, cy]
    if camera_model == "OPENCV_FISHEYE":
        params.extend([k1, k2, k3, k4])
    elif camera_model == "OPENCV":
        params.extend([k1, k2, p1, p2])
    elif camera_model == "PINHOLE":
        params.extend([0.0, 0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unknown camera model: {camera_model}")

    camera = Camera(
        width=width,
        height=height,
        model=camera_model,
        params=params,
    )
    cameras = [camera]
    images = []

    for i, frame in enumerate(data["frames"]):
        name = frame["file_path"]

        # Convert from nerfstudio to Colmap
        c2w = frame["transform_matrix"]
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1
        w2c = np.linalg.inv(c2w)

        image = Image(id=i, world_to_camera=w2c, camera_id=0, name=name)
        images.append(image)

    return cameras, images


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

    output_dir = cfg.get("output_dir")
    if output_dir is None:
        # default to data folder in data_root
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)

    render_devices = []
    if cfg.get("render_dslr", False):
        render_devices.append("dslr")
    if cfg.get("render_iphone", False):
        render_devices.append("iphone")

    # go through each scene
    for scene_id in tqdm(scene_ids, desc="scene"):
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
        render_engine = renderpy.Render()
        render_engine.setupMesh(str(scene.scan_mesh_path))
        for device in render_devices:
            if device == "dslr":
                cameras, images = read_nerfstudio_model(
                    scene.dslr_nerfstudio_transform_undistorted_path
                )
            else:
                cameras, images = read_nerfstudio_model(
                    scene.iphone_nerfstudio_transform_undistorted_path
                )
            assert len(cameras) == 1, "Multiple cameras not supported"
            camera = next(iter(cameras.values()))

            fx, fy, cx, cy = camera.params[:4]
            params = camera.params[4:]
            camera_model = camera.model
            render_engine.setupCamera(
                camera.height,
                camera.width,
                fx,
                fy,
                cx,
                cy,
                camera_model,
                params,  # Distortion parameters np.array([k1, k2, k3, k4]) or np.array([k1, k2, p1, p2])
            )

            near = cfg.get("near", 0.05)
            far = cfg.get("far", 20.0)
            rgb_dir = Path(cfg.output_dir) / scene_id / device / "render_rgb"
            depth_dir = Path(cfg.output_dir) / scene_id / device / "render_depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            for image_id, image in tqdm(images.items(), f"Rendering {device} images"):
                world_to_camera = image.world_to_camera
                rgb, depth, vert_indices = render_engine.renderAll(
                    world_to_camera, near, far
                )
                rgb = rgb.astype(np.uint8)
                # Make depth in mm and clip to fit 16-bit image
                depth = (
                    (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
                )
                imageio.imwrite(rgb_dir / image.name, rgb)
                depth_name = image.name.split(".")[0] + ".png"
                imageio.imwrite(depth_dir / depth_name, depth)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
