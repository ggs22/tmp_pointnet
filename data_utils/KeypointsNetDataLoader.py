import os
import json
import numpy as np
import open3d as o3d

from torch.utils.data import Dataset
from pathlib import Path


def pc_z_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(abs(pc), axis=0)
    pc = pc / m
    return pc


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class KeypointsDataset(Dataset):
    def __init__(self, root, npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = Path(root).joinpath('synsetoffset2category.txt')
        self.categories = {}
        self.normal_channel = normal_channel

        with open(str(self.catfile), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.categories[ls[0]] = ls[1]
        self.categories = {k: v for k, v in self.categories.items()}
        self.classes_original = dict(zip(self.categories, range(len(self.categories))))

        if class_choice is not None:
            self.categories = {k: v for k, v in self.categories.items() if k in class_choice}

        # Get the train, val & test filenames from the splits *.json files
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        # Make the split lists
        for category in self.categories:
            self.meta[category] = list()
            dir_point = os.path.join(self.root, self.categories[category])
            filenames = sorted(os.listdir(dir_point))
            if split == 'trainval':
                filenames = [filename for filename in filenames if ((filename[0:-4] in train_ids) or (filename[0:-4] in val_ids))]
            elif split == 'train':
                filenames = [filename for filename in filenames if filename[0:-4] in train_ids]
            elif split == 'val':
                filenames = [filename for filename in filenames if filename[0:-4] in val_ids]
            elif split == 'test':
                filenames = [filename for filename in filenames if filename[0:-4] in test_ids]
            else:
                print(f'Unknown split: {split}. Exiting..')
                exit(-1)

            for file in filenames:
                ply_path = str(Path(dir_point).joinpath(Path(file).stem + ".ply"))
                json_path = str(Path(dir_point).joinpath(Path(file).stem + ".json"))
                self.meta[category].append((ply_path, json_path))

        # Poplulate a list of file paths for this split
        self.datapath = list()
        for category in self.categories:
            for files_pair in self.meta[category]:
                self.datapath.append((category, files_pair))

        self.classes = dict()
        for category in self.categories.keys():
            self.classes[category] = self.classes_original[category]

        self.seg_classes = {'beam': [0, 1]}

        self.cache = dict()  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        # Caching strategy for data points
        if index in self.cache:
            point_set, cls, keypoints, sample_id = self.cache[index]
        else:
            ply_file_path = self.datapath[index][1][0]
            json_file_path = self.datapath[index][1][1]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            pcd = o3d.io.read_point_cloud(filename=ply_file_path)
            xyz = np.asarray(pcd.points)
            rgb = np.asarray(pcd.colors)
            data = np.concatenate([xyz, rgb], axis=1)
            point_set = data
            sample_id = json_file_path.split(os.path.sep)[-1:][0][0:-5:]
            with open(file=json_file_path, mode='r') as f:
                labelme_annotation = json.load(fp=f)
                keypoints = np.zeros(shape=(8, 5, 3))
                for ix, shape in enumerate(labelme_annotation['shapes']):
                    keypoints[ix] = np.array(shape['points'])

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, keypoints, sample_id)

        # Normalize the xyz data
        point_set = pc_z_normalize(point_set)

        # Random resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        return point_set, cls, keypoints, sample_id

    def __len__(self):
        return len(self.datapath)
