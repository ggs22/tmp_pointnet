#%%
""" Imports """
import numpy as np
import open3d as o3d
import argparse

from pathlib import Path
from tqdm import tqdm

source_dir = Path("C:\\Users\\LTI\\OneDrive - LTI\\Documents\\LTI\\Clients\\"
                  "TMS systems\\data\\Test bleu poutrelle 1 - 14 mars 2023\\Sans soudure\\ply files")
# source_dir = Path("C:\\Users\\LTI\\OneDrive - LTI\\Documents\\LTI\\Clients\\"
#                   "TMS systems\\data\\Test bleu poutrelle 2 - 14 mars 2023\\Avec crayon bleu\\ply files")
output_dir = source_dir.parent.joinpath('txt files')
output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir",
                        type=str,
                        required=True)
    parser.add_argument("--output_dir",
                        type=str,
                        required=True)

    return parser.parse_args()


def create_annotation_ouput(p: Path):
    """ Load the point cloud file """
    pcd_file_path = str(p)

    pcd = o3d.io.read_point_cloud(filename=pcd_file_path)

    ''' Put points and colors in a np.array '''
    points = np.asarray(pcd.points)

    ''' Lets flip the image so it is visualized in the same orientation than in Zivid studio '''
    points[:, 1] *= -1
    points[:, 2] *= -1

    ''' extract the colors '''
    colors = np.asarray(pcd.colors)
    r = colors[:, 0]
    g = colors[:, 1]
    b = colors[:, 2]

    r = colors[:, 0]
    g = colors[:, 1]
    b = colors[:, 2]

    ''' Get the dimensions limits '''
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    limits = np.array([xs.min(), xs.max(), ys.min(), ys.max(), zs.min(), zs.max()])

    ''' color separation '''
    # ix_r = np.where(r == 1)[0]
    ix_r = np.where(r < .4)[0]
    # ix_g = np.where(g == 1)[0]
    ix_g = np.where(g > 0.1)[0]
    # ix_b = np.intersect1d(np.where(b <= b.mean() + b.std())[0], np.where(b >= b.mean() - b.std())[0])
    # ix_b = np.intersect1d(np.where(b <= 0.66)[0], np.where(b >= 0.31)[0])
    ix_b = np.where(b >= 0.61)[0]
    # ix_b = np.where(b >= b.mean() - b.std())[0]
    ix_y = np.intersect1d(np.intersect1d(ix_r, ix_g), ix_b)

    ''' re-colorize annotation'''
    colors[ix_y] = [1, 0, 0]

    ''' re-assign pcd members '''
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(points)

    ''' create labels '''
    labels = np.zeros(shape=colors.shape[0])
    labels[ix_y] = 1

    ''' create txt file (as per the model current Dataset) '''
    output_file_path = output_dir.joinpath(str(p.stem)+'.txt')
    txt_points = np.concatenate([points, colors, labels.reshape((*labels.shape, 1))], axis=1)
    print(f'saving {str(output_file_path)} with {len(ix_y)} target points')
    np.savetxt(str(output_file_path), txt_points)

    ''' create segmentated point clouds '''
    output_file_path = output_dir.joinpath(str(p.stem)+'.ply')
    print(f'Writing ply file to : {str(output_file_path)}')
    o3d.io.write_point_cloud(filename=str(output_file_path), pointcloud=pcd)

    return pcd


if __name__ == "__main__":

    # args = parse_args()
    # source_dir = args.source_dir

    for ply_file in tqdm(source_dir.glob('*.ply')):
        create_annotation_ouput(p=ply_file, args=args)
