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

from pathlib import Path
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--source_dir",
                        help="The source directory where *.json and corresponding *.png & *.zdf file are located.",
                        required=True,
                        type=str)
    parser.add_argument("--down_sampling_factor",
                        help="Zivid downsampling 2by2 wil be applied down_smapling_factor of times. The keypoints "
                             "coordinates will be devided by 2^down_sampling_factor",
                        type=int,
                        default=3)
    parser.add_argument("--crop_margin",
                        help="The pixel margin left when cropping. Croping is done around annotations x, y extremums.",
                        type=int,
                        default=200)

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
            with(open(file=str(json_file))) as f:
                annotation = json.load(fp=f)

            with(open(file=str(zdf_file))) as f:
                frame = zivid.Frame(str(zdf_file))

            # Loop through keypoints and extract extremums of dimensions x, y (actually these are the pixel indexes)
            keypoints_extremums = dict()
            keypoints_extremums['x_min'], keypoints_extremums['x_max'] = np.inf, -np.inf
            keypoints_extremums['y_min'], keypoints_extremums['y_max'] = np.inf, -np.inf
            keypoints_extremums['z_min'], keypoints_extremums['z_max'] = np.inf, -np.inf
            for shape in annotation['shapes']:
                for point in shape['points']:
                    keypoints_extremums['x_min'] = min(point[0], keypoints_extremums['x_min'])
                    keypoints_extremums['x_max'] = max(point[0], keypoints_extremums['x_max'])
                    keypoints_extremums['y_min'] = min(point[1], keypoints_extremums['y_min'])
                    keypoints_extremums['y_max'] = max(point[1], keypoints_extremums['y_max'])

            margin = args.crop_margin
            keypoints_extremums['x_min'] = np.clip(a=int(np.floor(keypoints_extremums['x_min'])) - margin,
                                                   a_min=0,
                                                   a_max=frame.point_cloud().width)
            keypoints_extremums['x_max'] = np.clip(a=int(np.ceil(keypoints_extremums['x_max'])) + margin,
                                                   a_min=0,
                                                   a_max=frame.point_cloud().width)
            keypoints_extremums['y_min'] = np.clip(a=int(np.floor(keypoints_extremums['y_min'])) - margin,
                                                   a_min=0,
                                                   a_max=frame.point_cloud().height)
            keypoints_extremums['y_max'] = np.clip(a=int(np.ceil(keypoints_extremums['y_max'])) + margin,
                                                   a_min=0,
                                                   a_max=frame.point_cloud().height)

            croped_points_num = (keypoints_extremums['x_max'] - keypoints_extremums['x_min']) * (keypoints_extremums['y_max'] - keypoints_extremums['y_min'])
            print(f"Croped points: {croped_points_num}")

            # downsampling of the number of points
            factor = args.down_sampling_factor
            for _ in range(0, factor):
                frame.point_cloud().downsample(zivid.PointCloud.Downsampling.by2x2)
            points = frame.point_cloud().copy_data("xyz")
            rgb = frame.point_cloud().copy_data("rgba")[:, :, :3]

            # Loop through keypoints and scale their xy coordinates
            for shape_ix, shape in enumerate(annotation['shapes']):
                for point_ix, point in enumerate(shape['points']):
                    x, y = int(np.round(point[0])/(2**factor)), int(np.round(point[1])/(2**factor))
                    # colorize keypoint yellow
                    rgb[y, x, :] = [255, 255, 0]
                    # extract corresponding z annotations extremums
                    keypoints_extremums['z_min'] = min(points[y, x, 2], keypoints_extremums['z_min'])
                    keypoints_extremums['z_max'] = max(points[y, x, 2], keypoints_extremums['z_max'])
                    # we need to cast to float to avoid the json exportation bug
                    annotation['shapes'][shape_ix]['points'][point_ix] = [float(points[y, x, 1]),
                                                                          float(points[y, x, 0]),
                                                                          float(points[y, x, 2])]

            # scale extremums and crop the points cloud
            for key in keypoints_extremums:
                if 'z' not in key:  # we don't need to scale the z dimensions
                    keypoints_extremums[key] /= (2 ** factor)
                    keypoints_extremums[key] = int(keypoints_extremums[key])

            z_ixx = np.union1d(np.where(points[:, :, 2] < keypoints_extremums['z_max'])[1],
                               np.where(points[:, :, 2] > keypoints_extremums['z_min'])[1])
            z_ixy = np.union1d(np.where(points[:, :, 2] < keypoints_extremums['z_max'])[0],
                               np.where(points[:, :, 2] > keypoints_extremums['z_min'])[0])
            # TODO: (maybe) add z dimension croping

            points = points[keypoints_extremums['y_min']:keypoints_extremums['y_max'], keypoints_extremums['x_min']:keypoints_extremums['x_max'], :]
            rgb = rgb[keypoints_extremums['y_min']:keypoints_extremums['y_max'], keypoints_extremums['x_min']:keypoints_extremums['x_max'], :]

            # save annotation to ply file
            output_ply_path = output_dir.joinpath(json_file.stem + '.ply')
            pcd = o3d.geometry.PointCloud()

            points = points.reshape((points.shape[0] * points.shape[1], points.shape[2]))
            colors = rgb.reshape((rgb.shape[0] * rgb.shape[1], rgb.shape[2]))
            colors = colors/255

            # Trim off dataless points
            assert all(np.isnan(points[:, 0]) == np.isnan(points[:, 1]))
            assert all(np.isnan(points[:, 1]) == np.isnan(points[:, 2]))
            nan_ix = np.isnan(points[:, 0])

            points = points[~nan_ix, :]
            colors = colors[~nan_ix, :]

            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print(f"Saving {str(output_ply_path)} with {np.asarray(pcd.points).shape[0]} points")
            o3d.io.write_point_cloud(filename=str(output_ply_path), pointcloud=pcd)

            # Save a copy of the modified *.json annotation file with the resulting *.ply file
            with open(file=output_dir.joinpath(json_file.name), mode='w') as f:
                json.dump(obj=annotation, fp=f)
            # shutil.copy(src=json_file, dst=output_dir.joinpath(json_file.name))


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()
    main(args=args)
