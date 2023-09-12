import os


def make_sure_fold_exists(fold_path: str):
    if not os.path.exists(fold_path):
        os.mkdir(fold_path)
