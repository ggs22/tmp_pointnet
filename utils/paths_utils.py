import argparse

from pathlib import Path
import re

class PathUtils:

    def __init__(self,args: argparse.Namespace):
        self.experiment_output_dir = args.log_dir

    def get_root_dir(self) -> str:
        return str(Path(__file__).absolute().parent.parent)


    def get_data_dir(self) -> str:
        data_dir = str(Path(self.get_root_dir()).joinpath("data"))
        return data_dir


    def get_weld_kpts_dir(self) -> str:
        data_dir = self.get_data_dir()
        weld_kpts_dir = str(Path(data_dir).joinpath("poutrelle", "p1"))
        return weld_kpts_dir


    def get_current_model_path(self) -> str:
        checkpoint_path = Path(self.experiment_output_dir).joinpath('checkpoints')
        pattern = r'pointnet_model_epoch_([0-9]+).pth'
        latest_ix = -1
        res = str(checkpoint_path.joinpath('pointnet_model_epoch_1.pth'))
        for file_path in checkpoint_path.glob(pattern='*.pth'):
            match = re.match(pattern=pattern, string=str(file_path.name))
            if match is not None:
                ix = int(match[1])
                if ix > latest_ix:
                    latest_ix = ix
                    res = str(file_path)
        return res


    def get_best_validation_model_path(self) -> str:
        checkpoint_path = Path(self.experiment_output_dir).joinpath('checkpoints')
        pattern = r'pointnet_model_best_valid_epoch_([0-9]+).pth'
        latest_ix = -1
        res = str(checkpoint_path.joinpath('pointnet_model_best_valid_epoch_1.pth'))
        for file_path in checkpoint_path.glob(pattern='*.pth'):
            match = re.match(pattern=pattern, string=str(file_path.name))
            if match is not None:
                ix = int(match[1])
                if ix > latest_ix:
                    latest_ix = ix
                    res = str(file_path)
        return res

    def get_log_dir(self):



def test_suite(path_utils: PathUtils) -> None:
    print(path_utils.get_root_dir())
    print(path_utils.get_data_dir())


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("--test",
                        help="Test argument, debug only",
                        type=str,
                        default="test_arg")

    return parser.parse_args()

if __name__ == "__main__":
    pu = PathUtils(get_args())
    test_suite(path_utils=pu)
