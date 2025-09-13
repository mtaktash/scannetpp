import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common.scene_release import ScannetppScene_Release
from common.utils.utils import load_json, load_yaml_munch, read_txt_list, run_command


def compute_undistort_intrinsic(K, height, width, distortion_params):
    """
    K: (3,3) intrinsic matrix
    distortion_params: [k1, k2, p1, p2]
    """
    assert len(distortion_params.shape) == 1
    assert distortion_params.shape[0] == 4  # OPENCV has k1, k2, p1, p2

    # OpenCV wants (k1, k2, p1, p2[, k3, k4, k5, k6])
    distCoeffs = np.zeros(5)
    distCoeffs[:4] = distortion_params

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (width, height), alpha=0.0)
    # Make the cx and cy to be the center of the image
    new_K[0, 2] = width / 2.0
    new_K[1, 2] = height / 2.0

    return new_K


def undistort_frames(
    frames,
    K,
    height,
    width,
    distortion_params,
    input_image_dir,
    input_mask_dir,
    out_image_dir,
    out_mask_dir,
):
    new_K = compute_undistort_intrinsic(K, height, width, distortion_params)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, distortion_params, np.eye(3), new_K, (width, height), cv2.CV_32FC1
    )

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame["file_path"]
        image = cv2.imread(str(image_path))
        undistorted_image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        out_image_path = Path(out_image_dir) / frame["file_path"]
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_image_path), undistorted_image)

        # Mask
        mask_path = Path(input_mask_dir) / frame["mask_path"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if np.all(mask > 0):
            # No invalid pixels. Just use empty mask
            undistorted_mask = np.zeros((height, width), dtype=np.uint8) + 255
        else:
            undistorted_mask = cv2.remap(
                mask,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            # Filter the mask valid: 255, invalid: 0
            undistorted_mask[undistorted_mask < 255] = 0

        out_mask_path = Path(out_mask_dir) / frame["mask_path"]
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_mask_path), undistorted_mask)
    return new_K


def update_transforms_json(transforms, new_K, new_height, new_width):
    new_transforms = deepcopy(transforms)
    new_transforms["h"] = new_height
    new_transforms["w"] = new_width
    new_transforms["fl_x"] = new_K[0, 0]
    new_transforms["fl_y"] = new_K[1, 1]
    new_transforms["cx"] = new_K[0, 2]
    new_transforms["cy"] = new_K[1, 2]
    # The undistortion will be PINHOLE and have no distortion paramaters
    new_transforms["camera_model"] = "PINHOLE"
    for key in ("k1", "k2", "p1", "p2"):
        if key in new_transforms:
            new_transforms[key] = 0.0
    return new_transforms


def process_one_scene(scene_id, cfg):
    scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")
    input_image_dir = cfg.get("input_image_dir", None)
    if input_image_dir is None:
        input_image_dir = scene.iphone_rgb_dir
    else:
        input_image_dir = scene.iphone_data_dir / input_image_dir

    input_mask_dir = cfg.get("input_mask_dir", None)
    if input_mask_dir is None:
        input_mask_dir = scene.iphone_video_mask_dir
    else:
        input_mask_dir = scene.iphone_data_dir / input_mask_dir

    input_transforms_path = cfg.get("input_transforms_path", None)
    if input_transforms_path is None:
        input_transforms_path = scene.iphone_nerfstudio_transform_path
    else:
        input_transforms_path = scene.iphone_data_dir / input_transforms_path

    out_image_dir = scene.iphone_data_dir / cfg.out_image_dir
    out_mask_dir = scene.iphone_data_dir / cfg.out_mask_dir
    out_transforms_path = scene.iphone_data_dir / cfg.out_transforms_path

    is_compressed = False
    if not input_image_dir.exists() and not input_mask_dir.exists():
        is_compressed = True

        cmd = f"mkdir -p {input_image_dir} && tar -xf {scene.iphone_rgb_dir}.tar -C {input_image_dir}"
        run_command(cmd, verbose=True)

        cmd = f"mkdir -p {input_mask_dir} && tar -xf {scene.iphone_video_mask_dir}.tar -C {input_mask_dir}"
        run_command(cmd, verbose=True)

    transforms = load_json(input_transforms_path)
    assert len(transforms["frames"]) > 0
    frames = deepcopy(transforms["frames"])
    if "test_frames" not in transforms or len(transforms["test_frames"]) == 0:
        print(f"{scene_id} has no test split")
    elif not (input_image_dir / transforms["test_frames"][0]["file_path"]).exists():
        print(
            f"{scene_id} test image not found. Might due to the scene belonging to testing scenes. "
            "The resizing will skip those images."
        )
    else:
        assert len(transforms["test_frames"]) > 0
        frames += transforms["test_frames"]

    height = int(transforms["h"])
    width = int(transforms["w"])
    distortion_params = np.array(
        [
            float(transforms["k1"]),
            float(transforms["k2"]),
            float(transforms["p1"]),
            float(transforms["p2"]),
        ]
    )
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

    new_K = undistort_frames(
        frames,
        K,
        height,
        width,
        distortion_params,
        input_image_dir,
        input_mask_dir,
        out_image_dir,
        out_mask_dir,
    )

    new_trasforms = update_transforms_json(transforms, new_K, height, width)
    out_transforms_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_transforms_path, "w") as f:
        json.dump(new_trasforms, f, indent=4)

    if is_compressed:
        run_command(
            f"tar -cf {scene.iphone_data_dir}/rgb_undistorted.tar -C {out_image_dir} .",
            verbose=True,
        )
        run_command(
            f"tar -cf {scene.iphone_data_dir}/rgb_masks_undistorted.tar -C {out_mask_dir} .",
            verbose=True,
        )

    if is_compressed:
        print("Cleaning up...")
        run_command(f"rm -rf {input_image_dir}", verbose=True)
        run_command(f"rm -rf {input_mask_dir}", verbose=True)
        run_command(f"rm -rf {out_image_dir}", verbose=True)
        run_command(f"rm -rf {out_mask_dir}", verbose=True)


def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process
    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    elif cfg.get("splits"):
        scene_ids = []
        for split in cfg.splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    # get the options to process
    # go through each scene
    # for scene_id in tqdm(scene_ids, desc="scene"):
    #     process_one_scene(scene_id, cfg)

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_one_scene, sid, cfg): sid for sid in scene_ids
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="scene"):
            f.result()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
