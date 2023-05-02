from pathlib import Path


def get_root_dir() -> str:
    return str(Path(__file__).absolute().parent.parent)


def get_data_dir() -> str:
    data_dir = str(Path(get_root_dir()).joinpath("data"))
    return data_dir


def test_suite() -> None:
    print(get_root_dir())
    print(get_data_dir())


if __name__ == "__main__":
    test_suite()
