from pathlib import Path
import re

def get_root_dir() -> str:
    return str(Path(__file__).absolute().parent.parent)


def get_data_dir() -> str:
    data_dir = str(Path(get_root_dir()).joinpath("data"))
    return data_dir


def get_current_model_path(experiment_output_dir: str) -> str:
    checkpoint_path = Path(experiment_output_dir).joinpath('checkpoints')
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


def get_best_validation_model_path(experiment_output_dir: str) -> str:
    checkpoint_path = Path(experiment_output_dir).joinpath('checkpoints')
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

def test_suite() -> None:
    print(get_root_dir())
    print(get_data_dir())


if __name__ == "__main__":
    test_suite()
