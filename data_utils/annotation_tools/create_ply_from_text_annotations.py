import numpy
import numpy as np
import argparse
import open3d as o3d

from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir',
                        help='The source directory containing text annotation file (as per PointNet++ format)',
                        required=True)

    return parser.parse_args()


def create_ply_from_text_annotation(file_path: Path):
    arr = numpy.loadtxt(fname=str(file_path))
    pcd = o3d.geometry.PointCloud()
    if arr.shape[1] == 4:
        points = arr[:, 0:3]
        colors = np.ones(arr[:, 0:3].shape) * 0.5
    elif arr.shape[1] == 7:
        points = arr[:, 0:3]
        colors = arr[:, 3:6]
    else:
        raise RuntimeError("Data format not recognized!")

    t_ix = np.where(arr[:, -1:] == 1)[0]
    colors[t_ix] = [1, 0, 0]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    output_path = file_path.parent.joinpath(file_path.stem + '.ply')
    print(f'saving {str(output_path)} ({len(t_ix)} target points)')
    o3d.io.write_point_cloud(filename=str(output_path), pointcloud=pcd)


if __name__ == "__main__":
    args = parse_args()
    for file in Path(args.source_dir).glob(pattern='*.txt'):
        create_ply_from_text_annotation(file)
