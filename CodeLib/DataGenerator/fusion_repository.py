from typing import Any, List
import torch
import os


class FusionRepository:
    def __init__(self, root_path: str):
        self.root_path = root_path
        for _, dirs, _ in os.walk(root_path):
            self.fold_dirs = dirs
            break  # just need first level
        self.fold_dirs.sort()

    def get_idx_list(self) -> List[str]:
        return self.fold_dirs.copy()

    def get_fusion(self, idx: str) -> Any:
        # check idx
        if not (idx in self.fold_dirs):
            raise Exception("error model idx")
        paths = os.path.join(self.root_path, idx, "fe.pth")
        model = torch.load(paths)  # type:ignore
        return model  # type:ignore

    def get_matched_generaotrs(self, idx: str) -> List[str]:
        # check idx
        idx_list = idx.split("_")
        return idx_list
