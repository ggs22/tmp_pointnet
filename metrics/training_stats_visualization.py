import argparse
import re
import matplotlib.pyplot as plt

from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file_path",
                        help="The *.txt log file of the training.",
                        required=True)
    parser.add_argument("--output_dir",
                        help="The directory where the graph is going to be saved. If none is providied, the"
                             "graph will be saved in the same location than the log file.")

    return parser.parse_args()


def get_stats_from_log_file(log_file_path: str) -> dict:
    stages = ["train_dist_pattern",
              "train_loss_pattern",
              "test_dist_pattern",
              "test_loss_pattern",
              "epoch_pattern"]

    patterns = {"train_dist_pattern": "[Mm]ean train distance: (\d+.\d+)",
                "train_loss_pattern": "[Mm]ean train loss: (\d+.\d+)",
                "test_dist_pattern": ".* test Distance: (\d+.\d+).*",
                "test_loss_pattern": ".* test Loss: (\d+.\d+).*",
                "epoch_pattern": ".*Epoch: (\d+)"}

    train_dist_pattern = "[Mm]ean train distance: (\d+.\d+)"
    train_loss_pattern = "[Mm]ean train loss: (\d+.\d+)"
    test_pattern = ".*Epoch: (\d+).*test Distance: (\d+.\d+).*test Loss: (\d+.\d+),"

    stats = dict()
    for stage in stages:
        stats[stage.replace("_pattern", "")] = list()

    with open(file=log_file_path, mode='r') as f:
        lines = f.readlines()

    for line in lines:
        for stage, pattern in patterns.items():
            # match = re.match(pattern=pattern, string=line)
            # if match is not None:
            #     stats[stage.replace("_pattern", "")].append(float(match[1]))
            match = re.match(pattern=train_dist_pattern, string=line)
            if match is not None:
                stats['train_dist'].append(float(match[1]))
                break
            match = re.match(pattern=train_loss_pattern, string=line)
            if match is not None:
                stats['train_loss'].append(float(match[1]))
                break
            match = re.match(pattern=test_pattern, string=line)
            if match is not None:
                stats['epoch'].append(int(match[1]))
                stats['test_dist'].append(float(match[2]))
                stats['test_loss'].append(float(match[3]))
                break

    return stats


def save_stats_plat(stats_dict: dict, output_dir: str) -> None:
    output_path = Path(output_dir).joinpath("train_dist.png")
    lmin = min(len(stats_dict['epoch']), len(stats_dict['train_dist']))
    plt.plot(stats_dict['epoch'][0:lmin], stats_dict['train_dist'][0:lmin])
    plt.savefig(fname=str(output_path))
    plt.close()

    output_path = Path(output_dir).joinpath("test_dist.png")
    lmin = min(len(stats_dict['epoch']), len(stats_dict['test_dist']))
    plt.plot(stats_dict['epoch'][0:lmin], stats_dict['test_dist'][0:lmin])
    plt.savefig(fname=str(output_path))
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    log_file_path = args.log_file_path
    if not Path(log_file_path).exists():
        raise IOError(f"The file {log_file_path} does not exists!")
    if not Path(log_file_path).is_file() or Path(log_file_path).name[-4:] != '.txt':
        raise RuntimeError(f"The log file ({log_file_path}) needs to be a *.txt file!")
    if args.output_dir is None:
        output_dir = Path(log_file_path).parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir)

    stats_dict = get_stats_from_log_file(log_file_path=log_file_path)
    save_stats_plat(stats_dict=stats_dict, output_dir=output_dir)
