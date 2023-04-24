import json
import numpy
import numpy as np
import argparse
import open3d as o3d

from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir",
                        help='The source directory containing text annotation file (as per PointNet++ format)',
                        required=True)
    parser.add_argument("--output_dir",
                        help="The directory where the *.ply files will be saved. If none is provided, the files will"
                             "be saved in the source directory.")

    return parser.parse_args()


def create_ply_from_text_annotation(file_path: Path, json_file_path: Path) -> None:
    # extract xyz information (and color if available)
    print(f"Loading xyz information from {str(file_path)}")
    arr = numpy.loadtxt(fname=str(file_path))
    pcd = o3d.geometry.PointCloud()
    if arr.shape[1] == 4 or arr.shape[1] == 3:
        points = arr[:, 0:3]
        colors = np.ones(arr[:, 0:3].shape) * 0.5
    elif arr.shape[1] == 7 or arr.shape[1] == 6:
        points = arr[:, 0:3]
        colors = arr[:, 3:6]
    else:
        raise RuntimeError("Data format not recognized!")

    # extract keypoints prediction
    with open(file=str(json_file_path), mode='r') as f:
        keypoints_dict = json.load(fp=f)

    # TODO complet ply prediction generation
    for keypoint_ix, points in keypoints_dict.items():
        for point in points:
            x = float(point[1:-1:].split()[0])
            y = float(point[1:-1:].split()[1])

    # create and populate pointcloud object
    t_ix = np.where(arr[:, -1:] == 1)[0]
    colors[t_ix] = [1, 0, 0]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    output_path = file_path.parent.joinpath(file_path.stem + '.ply')

    # save to py file
    print(f'saving {str(output_path)} ({len(t_ix)} target points)')
    o3d.io.write_point_cloud(filename=str(output_path), pointcloud=pcd)


if __name__ == "__main__":
    args = parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise IOError(f"Directory {str(source_dir.absolute())} doesn't exists!")
    if not source_dir.is_dir():
        raise RuntimeError(f"The file {str(source_dir.absolute())} is not a directory!")
    if args.output_dir is None:
        output_dir = source_dir
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for xyz_txt_file in source_dir.glob(pattern='train_sample_*_xyz.txt'):

        # make sure corresponding json file with keypoints prediction exists
        json_file_path = xyz_txt_file.parent.joinpath(str(xyz_txt_file.name).replace("_xyz.txt", "_weld_paths.json"))
        if not json_file_path.exists():
            print(f"The JSON file corresponding to {str(xyz_txt_file)} has not been found!")
            continue
        # get the text file with the xyz information
        create_ply_from_text_annotation(xyz_txt_file, json_file_path)
