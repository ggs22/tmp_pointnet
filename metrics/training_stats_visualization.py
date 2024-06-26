import argparse
import re
import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file_path",
                        help="The *.txt log file of the PointNet2 training.",
                        required=True)
    parser.add_argument("--output_dir",
                        help="The directory where the graph is going to be saved. If none is providied, the"
                             "graph will be saved in the same location than the log file.")

    return parser.parse_args()


def get_stats_from_log_file(log_file_path: str) -> dict:
    stages = ["train_dist",
              "train_loss",
              "test_dist",
              "test_loss",
              "epoch"]

    train_dist_pattern = "[Mm]ean train distance: (\d+.\d+)"
    train_loss_pattern = "[Mm]ean train loss: (\d+.\d+)"
    test_pattern = ".*Epoch: (\d+).*test Distance: (\d+.\d+).*test Loss: (\d+.\d+),"

    stats = dict()
    for stage in stages:
        stats[stage] = list()

    with open(file=log_file_path, mode='r') as f:
        lines = f.readlines()

    for line in lines:
        match = re.match(pattern=train_dist_pattern, string=line)
        if match is not None:
            stats['train_dist'].append(float(match[1]))
            continue
        match = re.match(pattern=train_loss_pattern, string=line)
        if match is not None:
            stats['train_loss'].append(float(match[1]))
            continue
        match = re.match(pattern=test_pattern, string=line)
        if match is not None:
            stats['epoch'].append(int(match[1]))
            stats['test_dist'].append(float(match[2]))
            stats['test_loss'].append(float(match[3]))
            continue

    return stats


def save_stats_graphs(stats_dict: dict, output_dir: str, interval: list = None) -> None:

    def _plot_metric(epochs: list, metrics: list, metric_name: str, interval: str = None):
        output_path = Path(output_dir).joinpath(f"{metric_name}" +
                                                f"_{interval}" * (interval is not None) +
                                                ".png")
        lmin = min(len(epochs), len(metrics))
        plt.plot(epochs[0:lmin], metrics[0:lmin])
        print(f"Saving {output_path.name}")
        plt.xlabel("Époque")
        plt.ylabel(metric_name)
        plt.savefig(fname=str(output_path))
        plt.close()

    for stage, stats in stats_dict.items():
        if stage != "epoch":
            if interval is not None:
                min_bound = interval[0]
                max_bound = interval[1]

                if max_bound is not None:
                    _plot_metric(epochs=stats_dict['epoch'][min_bound:max_bound],
                                 metrics=stats[min_bound:max_bound],
                                 metric_name=stage,
                                 interval=f"{min_bound}_{max_bound}")
                else:
                    _plot_metric(epochs=stats_dict['epoch'][min_bound:],
                                 metrics=stats[min_bound:],
                                 metric_name=stage,
                                 interval=f"{min_bound}_")
            else:
                _plot_metric(epochs=stats_dict['epoch'],
                             metrics=stats,
                             metric_name=stage)


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
    save_stats_graphs(stats_dict=stats_dict, output_dir=output_dir, interval=[500, None])
    save_stats_graphs(stats_dict=stats_dict, output_dir=output_dir)
