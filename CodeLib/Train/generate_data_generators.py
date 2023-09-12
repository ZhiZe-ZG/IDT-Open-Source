from typing import Any, List
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np
from ..PLModel import GeneratorPLModule
from ..Tool import make_sure_fold_exists


def generate_generators(
    save_root: str,
    log_root:str,
    try_num: int = 100,
    train_data_num: int = 50000,
    test_data_num: int = 1000,
    batch_size: int = 500,
    max_epoch: int = 20,
    max_gpus: int = 1,
    acc_threshold: float = 0.9,
):
    # path
    make_sure_fold_exists(save_root)
    make_sure_fold_exists(log_root)
    # device
    device = "gpu" if torch.cuda.is_available() else "cpu"
    if device == "gpu":
        gpus = max_gpus if torch.cuda.is_available() else 0
    else:
        gpus = "auto"
    # train data
    sd = torch.rand(train_data_num, 1)
    dl = DataLoader(sd, batch_size=batch_size, shuffle=True)  # type: ignore
    # test data
    sd2 = torch.rand(test_data_num, 1)
    dl2 = DataLoader(sd2, batch_size=batch_size)  # type: ignore
    # super parameters
    for idx in range(try_num):
        print(f"Now Training {idx}")
        model = GeneratorPLModule()
        trainer = pl.Trainer(
            accelerator=device,
            devices=gpus,
            precision=16,
            limit_train_batches=1.0,
            max_epochs=max_epoch,
            default_root_dir=log_root,
        )
        trainer.fit(model, dl, dl2)  # type: ignore
        # test reversiility
        result_list: List[Any] = []
        z = None
        with torch.no_grad():
            for d in dl2:
                z = model(d)
                y = model.gd(z)
                dd = d.squeeze() > 0.5
                yy = y.squeeze().argmax(dim=1) > 0
                result_list.append(dd == yy)
        results = torch.concat(result_list)
        rate = results.sum() / test_data_num
        if rate > acc_threshold:
            save_dir = os.path.join(save_root, str(idx))
            make_sure_fold_exists(save_dir)
            # torch.save(model.ge.state_dict(), os.path.join(save_dir, "ge.pth"))  # type: ignore
            # torch.save(model.gd.state_dict(), os.path.join(save_dir, "gd.pth"))  # type: ignore
            torch.save(model.ge, os.path.join(save_dir, "ge.pth"))  # type: ignore
            torch.save(model.gd, os.path.join(save_dir, "gd.pth"))  # type: ignore
            # draw a image
            z: Any = z[:20]
            z = z.permute(0, 2, 3, 1)
            z = z.view(5, 4, 32, 32, 3)
            z = [[z[i][j] for j in range(z[i].shape[0])] for i in range(z.shape[0])]
            z = torch.concat([torch.concat(zz, dim=0) for zz in z], dim=1)
            z = z.data.numpy()
            z[z > 1] = 1
            z = (z * 255).astype(np.uint8)
            plt.imsave(os.path.join(save_dir, "sample.png"), z)  # type: ignore
