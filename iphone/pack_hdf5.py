import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from common.scene_release import ScannetppScene_Release
from common.utils.utils import load_json, load_yaml_munch, read_txt_list


def process_scene(scene):

    print("Loading the transforms...")
    transforms = load_json(scene.iphone_undistorted_nerfstudio_transform_path)

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

    frames = transforms["frames"]
    test_frames = transforms["test_frames"]

    temp_image_dir = scene.iphone_data_dir / "undistorted_images"
    temp_mask_dir = scene.iphone_data_dir / "undistorted_anon_masks"

    print("Uncompressing the images and masks...")
    os.system(
        f"mkdir -p {temp_image_dir} && tar -xf {scene.iphone_rgb_dir}.tar -C {temp_image_dir}"
    )
    os.system(
        f"mkdir -p {temp_mask_dir} && tar -xf {scene.iphone_video_mask_dir}.tar -C {temp_mask_dir}"
    )

    print("Packing the data into hdf5...")
    hdf5_path = scene.iphone_data_dir / f"data.hdf5"
    with h5py.File(hdf5_path, "w") as f:

        f.create_dataset("intrinsics", data=K)  # same for all images
        f.create_dataset(
            "frames", shape=(len(frames), height, width, 4), dtype=np.uint8
        )
        f.create_dataset("poses", shape=(len(frames), 4, 4), dtype=np.float32)
        f.create_dataset(
            "test_frames", shape=(len(test_frames), height, width, 4), dtype=np.uint8
        )
        f.create_dataset("test_poses", shape=(len(test_frames), 4, 4), dtype=np.float32)

        for split in ["train", "test"]:
            if split == "train":
                split_frames = frames
                split_dset_frames = f["frames"]
                split_dset_poses = f["poses"]
            else:
                split_frames = test_frames
                split_dset_frames = f["test_frames"]
                split_dset_poses = f["test_poses"]

            for i, frame in enumerate(split_frames):
                img_path = temp_image_dir / frame["file_path"]
                mask_path = temp_mask_dir / frame["file_path"]

                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                # H, W, 4
                img_rgba = np.concatenate([img, np.expand_dims(mask, axis=-1)], axis=-1)
                transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)

                split_dset_frames[i] = img_rgba
                split_dset_poses[i] = transform_matrix

    print(f"Packed data saved to {hdf5_path}")

    print("Cleaning up...")
    os.system(f"rm -rf {temp_image_dir}")
    os.system(f"rm -rf {temp_mask_dir}")


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
    for scene_id in tqdm(scene_ids, desc="scene"):
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")

        process_scene(scene)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
