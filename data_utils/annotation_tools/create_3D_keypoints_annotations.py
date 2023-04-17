"""
This script maps LabelMe keypoints (line strip) json file annotations made on 2D *.png file the corresponding *.zdf
file. Then is down sample the number of points and export the 3D keypoints annotations to a *.ply file.
"""

import zivid
import open3d as o3d
import numpy as np
import argparse
import json
import re
import shutil

from pathlib import Path
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--source_dir",
                        help="The source directory where *.json and correspond *.zdf file are located.",
                        required=True,
                        type=str)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    # Set up input/output directories
    source_dir: Path = Path(args.source_dir)
    output_dir: Path = source_dir.joinpath("keypoints_ply_files")
    output_dir.mkdir(parents=True, exist_ok=True)

    zivid.Application()  # needs to be instancied to load *.zdf files

    # Loop through json files in source directory
    for json_file in tqdm(source_dir.glob(pattern="*.json"), desc=f"Processing *.json files from {str(source_dir)}"):

        # make sure that we are working with the LabelMe *.json files
        json_fname_regex = re.compile(pattern="(\d{18}_.{8})\.json")
        match = json_fname_regex.match(string=str(json_file.name))
        if match is not None:

            # make sure that the corresponding *.zdf file exists
            zdf_file = Path(json_file.parent.joinpath(f"{match[1]}.zdf"))
            if not zdf_file.exists():
                raise IOError(f"The *.zdf file correspong to {str(json_file.name)} does not exists")

            # laod both files to workable objects
            annotation = dict()
            with(open(file=str(json_file))) as f:
                annotation = json.load(fp=f)

            # points: np.ndarray = None
            # rgb: np.ndarray = None
            with(open(file=str(zdf_file))) as f:
                frame = zivid.Frame(str(zdf_file))
                frame.point_cloud().downsample(zivid.PointCloud.Downsampling.by2x2)
                points = frame.point_cloud().copy_data("xyz")
                rgb = frame.point_cloud().copy_data("rgba")[:, :, :3]

            # Loop through keypoints and extract their xy coordinates
            for shape in annotation['shapes']:
                for point in shape['points']:
                    x, y = int(np.round(point[0])/2), int(np.round(point[1])/2)
                    rgb[y, x, :] = [255, 255, 0]  # colorize keypoint yellow

            # save annotation to ply file with a copy of the json file
            output_ply_path = output_dir.joinpath(json_file.stem + '.ply')
            pcd = o3d.geometry.PointCloud()
            nan_ix = np.isnan(points)
            points[nan_ix] = 0
            rgb[nan_ix] = 0
            points = points.reshape((points.shape[0] * points.shape[1], points.shape[2]))
            colors = rgb.reshape((rgb.shape[0] * rgb.shape[1], rgb.shape[2]))
            colors = colors/255

            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(filename=str(output_ply_path), pointcloud=pcd)
            shutil.copy(src=json_file, dst=output_dir.joinpath(json_file.name))


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    main(args=args)
