import zivid
import json
import numpy
import numpy as np
import argparse
import open3d as o3d

from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_pred_dir",
                        help='The source directory containing json keypoints annotations.',
                        required=True)
    parser.add_argument("--source_ori_dir",
                        help='The source directory containing original *.zdf files',
                        required=True)
    parser.add_argument("--output_dir",
                        help="The directory where the *.ply files will be saved. If none is provided, the files will"
                             "be saved in the source directory.")

    return parser.parse_args()


def get_predicted_keypoints(keypoints_prediction_path: Path) -> dict:
    with open(file=str(keypoints_prediction_path), mode='r') as f:
        keypoints = json.load(fp=f)
    return keypoints


def create_prediction_ply(originals_dir: Path, output_dir: Path, sample_id: str, keypoints: dict) -> None:
    for zdf_file in originals_dir.glob(pattern=f"**/{sample_id}.zdf"):
        frame = zivid.Frame(str(zdf_file))
        points = frame.point_cloud().copy_data(data_format="xyz")
        colors = frame.point_cloud().copy_data(data_format="rgba")[0:3]

        for weld_path_ix, points in keypoints.items():
            for point in points:
                x = int(np.round(point[0]))
                y = int(np.round(point[1]))
                colors[x, y, :] = [1., 0., 0.]  # Colorize predicted points in red

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        output_path = output_dir.joinpath(sample_id + "_predicted.ply")
        o3d.io.write_point_cloud(filename=output_path, pointcloud=pcd)

if __name__ == "__main__":
    # Input directories
    args = parse_args()
    source_dir = Path(args.source_pred_dir)
    ori_dir = Path(args.source_ori_dir)
    for path in [source_dir, ori_dir]:
        if not source_dir.exists():
            raise IOError(f"Directory {str(path.absolute())} doesn't exists!")
        if not source_dir.is_dir():
            raise RuntimeError(f"The file {str(path.absolute())} is not a directory!")

    # Output directory
    if args.output_dir is None:
        output_dir = source_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zivid.Application()  # (required to load Zivid frames)
    for json_prediction in source_dir.glob(pattern="*.json"):
        keypoins = get_predicted_keypoints(json_prediction)
        sample_id = json_prediction.stem.replace("_weld_paths", "")
        create_prediction_ply(originals_dir=ori_dir,
                              output_dir=output_dir,
                              sample_id=sample_id,
                              keypoints=keypoins)
