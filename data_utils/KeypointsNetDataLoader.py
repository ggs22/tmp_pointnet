import os
import json
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path


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

            for filename in filenames:
                path = str(Path(dir_point).joinpath(Path(filename).stem + ".txt"))
                self.meta[category].append(path)

        self.datapath = list()
        for category in self.categories:
            for filename in self.meta[category]:
                self.datapath.append((category, filename))

        self.classes = dict()
        for category in self.categories.keys():
            self.classes[category] = self.classes_original[category]

        self.seg_classes = {'beam': [0, 1]}

        self.cache = dict()  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        # Caching strategy for data points
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            file_path = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(file_path[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # choice = np.random.choice(len(seg), self.npoints, replace=True)
        # # resample
        # point_set = point_set[choice, :]
        # seg = seg[choice]

        if point_set.shape[0] > self.npoints:
            point_set = point_set[0:self.npoints]
            seg = seg[0:self.npoints]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)
