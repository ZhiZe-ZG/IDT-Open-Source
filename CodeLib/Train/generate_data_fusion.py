from typing import Any, List
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from ..Tool import make_sure_fold_exists
from ..DataGenerator import GeneratorRepository
from ..PLModel import ReversibleFusion


def generate_fusions(
    save_root: str,
    generator_path: str,
    log_root: str,
    generator_num: int = 4,
    try_num: int = 100,
    train_data_num: int = 10_000,
    test_data_num: int = 1_000,
    batch_size: int = 100,
    max_epoch: int = 20,
    max_gpus: int = 1,
    loss_threshold: float = 1e-2,
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
    # generator repository
    gr = GeneratorRepository(generator_path)
    gr_list = gr.get_idx_list()
    # super parameters
    for idx in range(try_num):
        print(f"Now Training {idx}")
        random.shuffle(gr_list)
        load_list = gr_list[:generator_num]
        generators = gr.get_generaotrs(load_list)
        # prepare data
        with torch.no_grad():
            td: List[torch.Tensor] = []
            for _ in range(train_data_num // batch_size):
                x = [torch.rand(batch_size, 1) for _ in range(generator_num)]
                images: List[torch.Tensor] = [
                    generators[id](x[id]) for id in range(generator_num)
                ]
                image: torch.Tensor = torch.concat(images, dim=1)
                td.append(image)
            tdd = [[t[i] for i in range(t.shape[0])] for t in td]
            del td
            tdd = sum(tdd, [])
            dl = DataLoader(tdd, batch_size=batch_size)  # type:ignore

            td2: List[torch.Tensor] = []
            for _ in range(test_data_num // batch_size):
                x = [torch.rand(batch_size, 1) for _ in range(generator_num)]
                images: List[torch.Tensor] = [
                    generators[id](x[id]) for id in range(generator_num)
                ]
                image: torch.Tensor = torch.concat(images, dim=1)
                td2.append(image)
            tdd2 = [[t[i] for i in range(t.shape[0])] for t in td2]
            del td2
            tdd2 = sum(tdd2, [])
            dl2 = DataLoader(tdd2, batch_size=100)  # type:ignore
        model = ReversibleFusion(fusion_num=generator_num)
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
        mean_loss = None
        d = None
        z = None
        y = None
        with torch.no_grad():
            for d in dl2:
                z = model(d)
                y = model.fd(z)
                loss = model.criterion(y, d)
                result_list.append(loss)
                mean_loss = sum(result_list) / len(result_list)  # type:ignore
        print("mean_loss", mean_loss)  # type:ignore
        if mean_loss < loss_threshold:  # type:ignore
            save_dir = os.path.join(save_root, "_".join(load_list))
            make_sure_fold_exists(save_dir)
            # save state dict will lead error in semantic
            torch.save(model.fe, os.path.join(save_dir, "fe.pth"))  # type: ignore
            torch.save(model.fd, os.path.join(save_dir, "fd.pth"))  # type: ignore
            # draw a image
            # just show 2 input
            input_images: List[torch.Tensor] = []
            for id in range(generator_num):
                input_images.append(d[:, id * 3 : (id + 1) * 3])  # type:ignore
            output_images: List[torch.Tensor] = []
            for id in range(generator_num):
                output_images.append(y[:, id * 3 : (id + 1) * 3])  # type:ignore
            image = torch.concat(
                input_images + [z] + output_images, dim=2  # type:ignore
            )  # type:ignore
            image_list = [image[i] for i in range(10)]
            show_image = torch.concat(image_list, dim=2)
            show_image = show_image.permute(1, 2, 0)
            show_image[show_image > 1] = 1
            show_image[show_image < 0] = 0
            # show_image = (show_image+1)/2
            show_image = show_image.data.numpy() * 255
            si = show_image.astype(np.uint8)
            plt.imsave(os.path.join(save_dir, "sample.png"), si)  # type: ignore
