"""
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
"""

import argparse
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imageio as iio
import lz4.block
import numpy as np
from tqdm import tqdm

from common.scene_release import ScannetppScene_Release
from common.utils.nerfstudio import prepare_transforms_json
from common.utils.utils import load_yaml_munch, read_txt_list, run_command


def extract_rgb(scene):
    scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -loglevel warning -i {scene.iphone_video_path} -start_number 0 -q:v 1 {scene.iphone_rgb_dir}/frame_%06d.jpg"
    run_command(cmd, verbose=True)

    frames = sorted(scene.iphone_rgb_dir.glob("frame_*.jpg"))
    frames = frames[::10]  # every 10th frame
    frames = [f.name for f in frames]

    # compress the extracted images
    cmd = f"tar -cf {scene.iphone_rgb_dir}.tar -C {scene.iphone_rgb_dir} " + " ".join(
        frames
    )
    run_command(cmd, verbose=True)


def extract_masks(scene):
    scene.iphone_video_mask_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -loglevel warning -i {str(scene.iphone_video_mask_path)} -pix_fmt gray -start_number 0 {scene.iphone_video_mask_dir}/frame_%06d.png"
    run_command(cmd, verbose=True)

    frames = sorted(scene.iphone_video_mask_dir.glob("frame_*.png"))
    frames = frames[::10]  # every 10th frame
    frames = [f.name for f in frames]

    cmd = (
        f"tar -cf {scene.iphone_video_mask_dir}.tar -C {scene.iphone_video_mask_dir} "
        + " ".join(frames)
    )
    run_command(cmd, verbose=True)


def extract_depth(scene):
    # global compression with zlib
    height, width = 192, 256
    sample_rate = 1
    scene.iphone_depth_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(scene.iphone_depth_path, "rb") as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(
            range(0, depth.shape[0], sample_rate), desc="decode_depth"
        ):
            iio.imwrite(
                f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png",
                (depth * 1000).astype(np.uint16),
            )
    # per frame compression with lz4/zlib
    except:
        frame_id = 0
        with open(scene.iphone_depth_path, "rb") as infile:
            while True:
                size = infile.read(4)  # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder="little")
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(
                        data, uncompressed_size=height * width * 2
                    )  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    depth = (depth * 1000).astype(np.uint16)

                # 6 digit frame id = 277 minute video at 60 fps
                iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", depth)
                frame_id += 1

    frames = sorted(scene.iphone_video_mask_dir.glob("frame_*.png"))
    frames = frames[::10]  # every 10th frame
    frames = [f.name for f in frames]

    cmd = (
        f"tar -cf {scene.iphone_depth_dir}.tar -C {scene.iphone_depth_dir} "
        + " ".join(frames)
    )
    run_command(cmd, verbose=True)


def cleanup_extracted(scene):
    cmd = f"rm -rf {scene.iphone_rgb_dir}"
    run_command(cmd, verbose=True)

    cmd = f"rm -rf {scene.iphone_video_mask_dir}"
    run_command(cmd, verbose=True)

    cmd = f"rm -rf {scene.iphone_depth_dir}"
    run_command(cmd, verbose=True)


def process_one_scene(scene_id, cfg):
    scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")

    if cfg.extract_rgb:
        extract_rgb(scene)

        if cfg.extract_nerfstudio_transforms:

            # every 10th image as train (as in transforms)
            train_list = sorted(scene.iphone_rgb_dir.glob("*.jpg"))
            train_list = [x.name for x in train_list]
            train_list = train_list[::10]

            prepare_transforms_json(
                model_path=scene.iphone_colmap_dir,
                out_path=scene.iphone_nerfstudio_transform_path,
                train_list=train_list,
                test_list=[],
                has_mask=True,
            )

    if cfg.extract_masks:
        extract_masks(scene)

    if cfg.extract_depth:
        extract_depth(scene)

    if cfg.cleanup_extracted:
        cleanup_extracted(scene)


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

    # get the options to process
    # go through each scene
    # for scene_id in tqdm(scene_ids, desc="scene"):
    #     process_one_scene(scene_id, cfg)

    with ProcessPoolExecutor(max_workers=16) as executor:
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
