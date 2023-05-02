import zivid
import json
import argparse
import re
import numpy as np
import open3d as o3d

from tqdm import tqdm
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


def create_prediction_ply(originals_dir: Path,
                          output_dir: Path,
                          sample_id: str,
                          stage: str,
                          keypoints: dict) -> None:

    # We process one sample at a time
    for zdf_file in originals_dir.glob(pattern=f"**/{sample_id}.zdf"):
        frame = zivid.Frame(str(zdf_file))
        points = frame.point_cloud().copy_data(data_format="xyz")
        colors = frame.point_cloud().copy_data(data_format="rgba")[:, :, 0:3]

        # For each welding path, extract each predicted keypoint
        for weld_path_ix, pred_points in keypoints.items():
            for point in pred_points:
                x = int(np.round(point[0]))
                y = int(np.round(point[1]))

                x = np.clip(x, 0, points.shape[1] - 1)
                y = np.clip(y, 0, points.shape[0] - 1)
                width = 10
                colors[y-width:y+width, x-width:x+width, :] = [255, 0., 0.]  # Colorize predicted points in red

        # create open3D objects
        points = np.reshape(points, newshape=(points.shape[0] * points.shape[1], points.shape[2]))
        colors = np.reshape(colors, newshape=(colors.shape[0] * colors.shape[1], colors.shape[2]))
        colors = np.divide(colors, 255)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save to ply file
        output_path = output_dir.joinpath(sample_id + f"_{stage}_predicted.ply")
        o3d.io.write_point_cloud(filename=str(output_path), pointcloud=pcd)


def extract_sample_id(json_file_name: Path) -> str:
    pattern = r"(\d+_\w+)_(train|test).*\.json"
    match = re.match(pattern=pattern, string=json_file_name.name)
    if match is None:
        raise RuntimeError(f"{str(json_file_name)} is not a recognized keypoints prediction file!")
    else:
        return match[1], match[2]


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

    # Process every JSON predictions file
    for json_prediction in tqdm(source_dir.glob(pattern="*.json"), desc="Process JSON -> PLY predctions"):
        keypoins = get_predicted_keypoints(json_prediction)
        sample_id, stage = extract_sample_id(json_prediction)
        create_prediction_ply(originals_dir=ori_dir,
                              output_dir=output_dir,
                              sample_id=sample_id,
                              stage=stage,
                              keypoints=keypoins)
