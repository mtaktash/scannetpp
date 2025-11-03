import argparse
from pathlib import Path

import torch.multiprocessing as mp

from common.scene_release import ScannetppScene_Release
from common.utils.utils import load_yaml_munch
from planar.ground_truth_planes import process_scene_planar_mesh


def main(args):
    cfg = load_yaml_munch(args.config_file)
    scene_id = "7b6477cb95"

    scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root))
    process_scene_planar_mesh(scene)


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
