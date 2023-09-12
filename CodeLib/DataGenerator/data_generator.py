from typing import List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

num_workers = 0

def generate_dataloader(
    ml: List[nn.Module],
    mf: nn.Module,
    train_num: int = 500,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test:int = 100,
) -> Any:
    # prepare data
    with torch.no_grad():
        feature_num = len(ml)
        td = []
        ll = []
        for _ in range(train_num // batch_size):
            x = [torch.rand(batch_size, 1) for _ in ml]
            lx = torch.concat(x, dim=1)
            bool_matrix = lx >= 0.5
            bit_matrix = torch.bitwise_left_shift(
                bool_matrix, torch.arange(feature_num)
            )
            labels = torch.sum(bit_matrix, dim=1)
            ll.append(labels)  # type:ignore
            images = [m(dx) for m, dx in zip(ml, x)]
            images = torch.concat(images, dim=1)
            td.append(images)  # type:ignore
        td = [mf(d) for d in td]  # type:ignore
        td = [[t[i] for i in range(t.shape[0])] for t in td]
        td = sum(td, [])
        ll = [[t[i] for i in range(t.shape[0])] for t in ll]  # type:ignore
        ll = sum(ll, [])  # type:ignore
        dl = [d for d in zip(td, ll)]  # type:ignore
        dl = DataLoader(dl, batch_size=batch_size,num_workers=num_workers)  # type:ignore

        td2 = []
        ll2 = []
        for _ in range(test_num // batch_size):
            x = [torch.rand(batch_size, 1) for _ in ml]
            lx = torch.concat(x, dim=1)
            bool_matrix = lx >= 0.5
            bit_matrix = torch.bitwise_left_shift(
                bool_matrix, torch.arange(feature_num)
            )
            labels = torch.sum(bit_matrix, dim=1)
            ll2.append(labels)  # type:ignore
            images = [m(dx) for m, dx in zip(ml, x)]
            images = torch.concat(images, dim=1)
            td2.append(images)  # type:ignore
        td2 = [mf(d) for d in td2]  # type:ignore
        td2 = [[t[i] for i in range(t.shape[0])] for t in td2]
        td2 = sum(td2, [])
        ll2 = [[t[i] for i in range(t.shape[0])] for t in ll2]  # type:ignore
        ll2 = sum(ll2, [])  # type:ignore
        dl2 = [d for d in zip(td2, ll2)]  # type:ignore
        dl2 = DataLoader(dl2, batch_size=batch_size_test,num_workers=num_workers)  # type:ignore
        return dl, dl2

def generate_dataloader_dataset(
    ds1:Any,
    ds2:Any,
    train_num: int = 500,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test:int = 100,
) -> Any:
    # ds1i = [x for x in RandomSampler(ds1, num_samples=train_num)]
    # ds2i = [x for x in RandomSampler(ds2, num_samples=test_num)]
    dl = DataLoader(ds1, batch_size=batch_size,num_workers=num_workers)  # type:ignore
    dl2 = DataLoader(ds2, batch_size=batch_size_test,num_workers=num_workers)  # type:ignore
    return dl, dl2
