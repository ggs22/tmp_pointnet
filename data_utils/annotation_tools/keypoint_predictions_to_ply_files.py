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


def get_target_keypoints(originals_dir: Path, sample_id: str) -> dict:
    json_file_path = [file for file in originals_dir.glob(pattern=f"**/keypoints_ply_files/{sample_id}.json")][0]
    with open(file=str(json_file_path), mode='r') as f:
        target = json.load(fp=f)

    return target


def create_prediction_ply(originals_dir: Path,
                          output_dir: Path,
                          sample_id: str,
                          stage: str,
                          prediction: dict,
                          target: dict) -> None:

    # We process one sample at a time
    zdf_file = [file for file in originals_dir.glob(pattern=f"**/{sample_id}.zdf")][0]

    # for zdf_file in :
    frame = zivid.Frame(str(zdf_file))
    points = frame.point_cloud().copy_data(data_format="xyz")
    colors = frame.point_cloud().copy_data(data_format="rgba")[:, :, 0:3]

    # create open3D objects
    points = np.reshape(points, newshape=(points.shape[0] * points.shape[1], points.shape[2]))
    colors = np.reshape(colors, newshape=(colors.shape[0] * colors.shape[1], colors.shape[2]))
    colors = np.divide(colors, 255)


    # For each welding path, extract each predicted keypoint
    for weld_path_ix, pred_points in tqdm(prediction.items()):
        for point in pred_points:
            x = point[0]
            y = point[1]
            z = point[2]

            width = 5
            # add a prediction point as a red 3D cross
            for i in range(int(-width/2), int(width/2)):
                points = np.concatenate([points, np.array([[y + i, x, z]])], axis=0)
                points = np.concatenate([points, np.array([[y, x + i, z]])], axis=0)
                points = np.concatenate([points, np.array([[y, x, z + i]])], axis=0)
                colors = np.concatenate([colors, np.array([[1, 0, 0]])], axis=0)
                colors = np.concatenate([colors, np.array([[1, 0, 0]])], axis=0)
                colors = np.concatenate([colors, np.array([[1, 0, 0]])], axis=0)

    for weld_path in tqdm(target['shapes']):
        for point in weld_path['points']:
            x = point[0]
            y = point[1]
            z = point[2]

            width = 5
            # add a target point as a green 3D cross
            for i in range(int(-width / 2), int(width / 2)):
                points = np.concatenate([points, np.array([[y + i, x, z]])], axis=0)
                points = np.concatenate([points, np.array([[y, x + i, z]])], axis=0)
                points = np.concatenate([points, np.array([[y, x, z + i]])], axis=0)
                colors = np.concatenate([colors, np.array([[0, 1, 0]])], axis=0)
                colors = np.concatenate([colors, np.array([[0, 1, 0]])], axis=0)
                colors = np.concatenate([colors, np.array([[0, 1, 0]])], axis=0)

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
    json_predicions_files = [file for file in source_dir.glob(pattern="*.json")]
    for json_prediction in tqdm(json_predicions_files, desc="Process JSON -> PLY predctions"):

        keypoins = get_predicted_keypoints(json_prediction)
        sample_id, stage = extract_sample_id(json_prediction)
        target = get_target_keypoints(originals_dir=ori_dir, sample_id=sample_id)

        create_prediction_ply(originals_dir=ori_dir,
                              output_dir=output_dir,
                              sample_id=sample_id,
                              stage=stage,
                              prediction=keypoins,
                              target=target)
