import os
from pathlib import Path
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Split txt files into train, validation, and test sets.')
    parser.add_argument('--input_dir',
                        type=str, help='Input directory to search for txt files', required=True)
    parser.add_argument('--output_dir',
                        type=str, help='Input directory to search for txt files')
    parser.add_argument('--exit',
                        type=str, default='.txt', help='extension of the file type to be split')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of files to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of files to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Ratio of files to use for testing')
    return parser.parse_args()


def get_files_list(input_dir):
    files_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                files_list.append(os.path.join(root, file))
    return files_list


def split_files(files_list, train_ratio, val_ratio, test_ratio):
    random.shuffle(files_list)
    train_size = int(train_ratio * len(files_list))
    val_size = int(val_ratio * len(files_list))
    test_size = int(test_ratio * len(files_list))
    assert train_size + val_size + test_size == len(files_list)  # Make sure we use all the files
    train_files = files_list[:train_size]
    val_files = files_list[train_size:train_size+val_size]
    test_files = files_list[train_size+val_size:train_size+val_size+test_size]
    return train_files, val_files, test_files


def write_files_to_json(files, list_name):
    if args.output_dir is None:
        output_dir = args.input_dir
    else:
        output_dir = args.output_dir
    output_path = Path(output_dir).joinpath(f'shuffled_{list_name}_file_list.json')

    with open(str(output_path), 'w') as f:
        f.write('[')
        for ix, file in enumerate(files):
            f.write(f'\"foo/bar/{Path(file).stem}\"' + ',' * (ix < len(files) - 1))
        f.write(']')


if __name__ == '__main__':
    args = parse_args()
    input_dir = args.input_dir
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio

    files_list = get_files_list(input_dir)
    train_files, val_files, test_files = split_files(files_list, train_ratio, val_ratio, test_ratio)

    print(f"Found {len(files_list)} txt files in directory {input_dir}")
    print(f"Splitting files into {len(train_files)} training files, {len(val_files)} validation files, and {len(test_files)} testing files")

    write_files_to_json(train_files, 'train')
    write_files_to_json(val_files, 'val')
    write_files_to_json(test_files, 'test')
