from typing import Any, List
import torch
import os

class GeneratorRepository:
    def __init__(self, root_path: str):
        self.root_path = root_path
        for _, dirs, _ in os.walk(root_path):
            self.fold_dirs = dirs
            break  # just need first level
        self.fold_dirs.sort()

    def get_idx_list(self) -> List[str]:
        return self.fold_dirs.copy()

    def get_generaotrs(self, idx_list: List[str]) -> Any:
        # check idx
        if not all([idx in self.fold_dirs for idx in idx_list]):
            raise Exception("error model idx")
        paths = [os.path.join(self.root_path, p, "ge.pth") for p in idx_list]
        models = [torch.load(p) for p in paths]  # type:ignore
        # models = [GenerateEncoder() for _ in paths]
        # _ = [
        #     GenerateEncoder().load_state_dict(torch.load(p))  # type:ignore
        #     for p in paths
        # ]
        return models  # type:ignore
