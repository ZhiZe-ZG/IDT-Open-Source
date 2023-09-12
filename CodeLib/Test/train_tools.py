from typing import Dict, Any
import random
import torch
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score  # type:ignore
from ..DataGenerator import generate_dataloader, generate_dataloader_dataset
from ..DataGenerator import GeneratorRepository
from ..DataGenerator import FusionRepository
from ..PLModel import OverAllModelPL, OverAllVGGPL, OverAllRESPL
from ..PLModel import SeperateModelPL, SeperateModelSPL
from torch.utils.data import RandomSampler

# import torchvision


def train_model(
    model: pl.LightningModule,
    train_loader: DataLoader,  # type:ignore
    test_loader: DataLoader,  # type:ignore
    max_epochs: int = 20,
    gpus: int = 1,
):
    #     from pytorch_lightning.callbacks import ModelCheckpoint
    # checkpoint_callback = ModelCheckpoint(
    # #     monitor='test_IoU_sum',
    #     monitor=None,
    #     dirpath='./lightning_logs/OverAllModel',
    #     filename='OverAllModel-{epoch:02d}' , #{test_IoU_one_0:.2f}
    # #     save_top_k=10,
    #     mode='min',
    # )
    trainer = pl.Trainer(
        # profiler="simple",# test
        devices=gpus,
        precision=16,
        limit_train_batches=1.0,
        max_epochs=max_epochs,
        enable_checkpointing=False,  # close checkpoint to save time
        # callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, test_loader)  # type:ignore


def test_acc(
    model: pl.LightningModule,
    test_loader: DataLoader,  # type:ignore
) -> float:
    pre_label_list = []
    label_list = []
    for image, label in test_loader:
        # pre = model(image)
        # pre_label = torch.argmax(pre, dim=1)
        pre_label = model(image)
        pre_label_list.append(pre_label)  # type:ignore
        label_list.append(label)  # type:ignore
    pre_label = torch.concat(pre_label_list).data.numpy()
    label = torch.concat(label_list).data.numpy()
    acc: float = accuracy_score(label, pre_label)  # type:ignore
    return acc


# def test_acc_seperate(
#     model: pl.LightningModule,
#     test_loader: DataLoader,  # type:ignore
# ) -> float:
#     '''
#     for seperated model
#     '''
#     pre_label_list = []
#     label_list = []
#     for image,label in test_loader:
#         pre = model(image)
#         pre = torch.stack(pre).argmax(dim=2)
#         pre = model.label_mask[:,None]*pre # type:ignore
#         pre = pre.sum(dim=0) # type:ignore
#         pre_label_list.append(pre) # type:ignore
#         label_list.append(label) # type:ignore
#     pre_label = torch.concat(pre_label_list).data.numpy()
#     label = torch.concat(label_list).data.numpy()
#     acc: float = accuracy_score(label, pre_label)  # type:ignore
#     return acc


def get_one_point(
    gr: GeneratorRepository,
    gf: FusionRepository,
    fusion_idx: str,
    model: pl.LightningModule,
    train_num: int = 2000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> float:
    """
    train a model and return test acc
    """
    mf = gf.get_fusion(fusion_idx)
    ml = gr.get_generaotrs(gf.get_matched_generaotrs(fusion_idx))
    dl, dl2 = generate_dataloader(
        ml,
        mf,
        train_num=train_num,
        test_num=test_num,
        batch_size=batch_size,
        batch_size_test=batch_size_test,
    )
    train_model(model, dl, dl2, max_epochs=epochs, gpus=gpus)
    acc = test_acc(model, test_loader=dl2)
    return acc

# handle data list
def preprocess_data_one_MNIST(data_pair, label_trans):
    x,y = data_pair
    x = torch.concat([x,x,x],dim=0) # 3-channel
    y = label_trans.index(y)
    return (x,y)

def preprocess_data_one_CIFAR(data_pair, label_trans):
    x,y = data_pair
    # x = torch.concat([x,x,x],dim=0) # 3-channel
    y = label_trans.index(y)
    return (x,y)


def mnist_dataset_pair():
    ds1 = torchvision.datasets.MNIST(
            "./data/",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            train=True,
            download=True,
        )
    ds2 = torchvision.datasets.MNIST(
        "./data/",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
            ]
        ),
        train=False,
        download=True,
    )
    label_trans = [-1 for i in range(2**4)]
    label_trans[0b0010] = 3
    label_trans[0b0011] = 7
    label_trans[0b0100] = 0
    label_trans[0b0101] = 1
    label_trans[0b0110] = 2
    label_trans[0b1000] = 6
    label_trans[0b1001] = 9
    label_trans[0b1010] = 4
    label_trans[0b1100] = 8
    label_trans[0b1110] = 5
    ds1 = [preprocess_data_one_MNIST(d, label_trans) for d in ds1]
    ds2 = [preprocess_data_one_MNIST(d, label_trans) for d in ds2]
    return (ds1, ds2)

def fmnist_dataset_pair():
    ds1 = torchvision.datasets.FashionMNIST(
            "./data/",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            train=True,
            download=True,
        )
    ds2 = torchvision.datasets.FashionMNIST(
        "./data/",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
            ]
        ),
        train=False,
        download=True,
    )
    label_trans = [-1 for i in range(2**4)]
    label_trans[0b0000] = 4
    label_trans[0b0001] = 2
    label_trans[0b0011] = 6
    label_trans[0b0100] = 0
    label_trans[0b0101] = 3
    label_trans[0b1010] = 9
    label_trans[0b1011] = 1
    label_trans[0b1100] = 8
    label_trans[0b1101] = 7
    label_trans[0b1111] = 5
    ds1 = [preprocess_data_one_MNIST(d, label_trans) for d in ds1]
    ds2 = [preprocess_data_one_MNIST(d, label_trans) for d in ds2]
    return (ds1, ds2)

def cifar_dataset_pair():
    ds1 = torchvision.datasets.CIFAR10(
            "./data/",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            train=True,
            download=True,
        )
    ds2 = torchvision.datasets.CIFAR10(
        "./data/",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
            ]
        ),
        train=False,
        download=True,
    )
    label_trans = [-1 for i in range(2**4)]
    label_trans[0b0000] = 0
    label_trans[0b0010] = 8
    label_trans[0b0100] = 1
    label_trans[0b0110] = 9
    label_trans[0b1000] = 6
    label_trans[0b1001] = 2
    label_trans[0b1100] = 5
    label_trans[0b1101] = 3
    label_trans[0b1110] = 7
    label_trans[0b1111] = 4
    ds1 = [preprocess_data_one_CIFAR(d, label_trans) for d in ds1]
    ds2 = [preprocess_data_one_CIFAR(d, label_trans) for d in ds2]
    return (ds1, ds2)

def get_one_point_dataset(
    ds: Any,
    fusion_idx: str,
    model: pl.LightningModule,
    train_num: int = 2000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> float:
    """
    train a model and return test acc
    """
    ds1, ds2 = ds
    random.shuffle(ds1)
    ds1 = ds1[:train_num]
    random.shuffle(ds2)
    ds2 = ds2[:test_num]
    dl, dl2 = generate_dataloader_dataset(
        ds1,  # type:ignore
        ds2,  # type:ignore
        train_num=train_num,
        test_num=test_num,
        batch_size=batch_size,
        batch_size_test=batch_size_test,
    )
    train_model(model, dl, dl2, max_epochs=epochs, gpus=gpus)
    acc = test_acc(model, test_loader=dl2)
    return acc


def generate_allinone_model(feature_num: int = 2):
    return OverAllModelPL(class_num=2**feature_num)


def generate_allinone_vgg16(feature_num: int = 2):
    return OverAllVGGPL(class_num=2**feature_num)


def generate_allinone_res(feature_num: int = 2):
    return OverAllRESPL(class_num=2**feature_num)


def generate_seperate_model(feature_num: int = 2):
    return SeperateModelPL(feature_num=feature_num)


def generate_seperateS_model(feature_num: int = 2):
    return SeperateModelSPL(feature_num=feature_num)


def get_statis_point(
    gr: GeneratorRepository,
    gf: FusionRepository,
    fusion_idx: str,
    feature_num: int,
    model_f: Any,
    try_times: int = 10,
    train_num: int = 2000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> Dict[str, Any]:
    # model = model_f(feature_num)
    results = [
        get_one_point(
            gr,
            gf,
            fusion_idx,
            model_f(feature_num),
            train_num=train_num,
            test_num=test_num,
            batch_size=batch_size,
            batch_size_test=batch_size_test,
            epochs=epochs,
            gpus=gpus,
        )
        for _ in range(try_times)
    ]
    results = torch.Tensor(results)
    d: Dict[str, Any] = {}
    d["mean"] = float(results.mean().data)
    d["max"] = float(results.max().data)  # type:ignore
    d["min"] = float(results.min().data)  # type:ignore
    d["results"] = results
    return d


def get_statis_point_dataset(
    ds: Any,
    fusion_idx: str,
    feature_num: int,
    model_f: Any,
    try_times: int = 10,
    train_num: int = 2000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> Dict[str, Any]:
    # model = model_f(feature_num)
    results = [
        get_one_point_dataset(
            ds,
            fusion_idx,
            model_f(feature_num),
            train_num=train_num,
            test_num=test_num,
            batch_size=batch_size,
            batch_size_test=batch_size_test,
            epochs=epochs,
            gpus=gpus,
        )
        for _ in range(try_times)
    ]
    results = torch.Tensor(results)
    d: Dict[str, Any] = {}
    d["mean"] = float(results.mean().data)
    d["max"] = float(results.max().data)  # type:ignore
    d["min"] = float(results.min().data)  # type:ignore
    d["results"] = results
    return d


def try_all_generators(
    gr: GeneratorRepository,
    gf: FusionRepository,
    feature_num: int,
    model_f: Any,
    try_times: int = 5,
    train_num: int = 1000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> Dict[str, Any]:
    gf_list = [n for n in gf.get_idx_list() if len(n.split("_")) == feature_num]
    results = [
        get_statis_point(
            gr,
            gf,
            fusion_idx,
            feature_num,
            model_f,
            try_times=try_times,
            train_num=train_num,
            test_num=test_num,
            batch_size=batch_size,
            batch_size_test=batch_size_test,
            epochs=epochs,
            gpus=gpus,
        )
        for fusion_idx in gf_list
    ]
    results = [r["mean"] for r in results]
    results = torch.Tensor(results)
    d: Dict[str, Any] = {}
    d["mean"] = float(results.mean().data)
    d["max"] = float(results.max().data)  # type:ignore
    d["min"] = float(results.min().data)  # type:ignore
    d["results"] = results
    return d


def try_all_generators_dataset(
    ds: str,
    feature_num: int,
    model_f: Any,
    try_times: int = 5,
    train_num: int = 1000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> Dict[str, Any]:
    if ds == "MNIST":
        # MINST Data
        dst = mnist_dataset_pair()
    elif ds == 'FMNIST':
        # Fashion MNIST
        dst = fmnist_dataset_pair()
    elif ds == 'CIFAR':
        # CIFAR 10
        dst = cifar_dataset_pair()
    results = [
        get_statis_point_dataset(
            dst, # type:ignore
            fusion_idx,
            feature_num,
            model_f,
            try_times=try_times,
            train_num=train_num,
            test_num=test_num,
            batch_size=batch_size,
            batch_size_test=batch_size_test,
            epochs=epochs,
            gpus=gpus,
        )
        for fusion_idx in ["TestOneTime"]
    ]
    results = [r["mean"] for r in results]
    results = torch.Tensor(results)
    d: Dict[str, Any] = {}
    d["mean"] = float(results.mean().data)
    d["max"] = float(results.max().data)  # type:ignore
    d["min"] = float(results.min().data)  # type:ignore
    d["results"] = results
    return d


def get_statis_point_choosebest(
    gr: GeneratorRepository,
    gf: FusionRepository,
    fusion_idx: str,
    feature_num: int,
    model_f: Any,
    try_times: int = 10,
    train_num: int = 2000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> Dict[str, Any]:
    mf = gf.get_fusion(fusion_idx)
    ml = gr.get_generaotrs(gf.get_matched_generaotrs(fusion_idx))
    save_model = None
    for try_idx in range(try_times):
        model = model_f(feature_num)
        dl, dl2 = generate_dataloader(
            ml,
            mf,
            train_num=train_num,
            test_num=test_num,
            batch_size=batch_size,
            batch_size_test=batch_size_test,
        )
        train_model(model, dl, dl2, max_epochs=epochs, gpus=gpus)
        if save_model is None:
            save_model = model
        else:
            for part_idx in range(len(save_model.loss_list)):
                if model.loss_list[part_idx] < save_model.loss_list[part_idx]:
                    save_model.decoder_list[part_idx] = model.decoder_list[part_idx]
    acc = test_acc(save_model, test_loader=dl2)  # type:ignore
    ##
    results = [
        acc,
    ]
    results = torch.Tensor(results)
    d: Dict[str, Any] = {}
    d["mean"] = float(results.mean().data)
    d["max"] = float(results.max().data)  # type:ignore
    d["min"] = float(results.min().data)  # type:ignore
    d["results"] = results
    return d


def get_statis_point_choosebest_dataset(
    ds: Any,
    fusion_idx: str,
    feature_num: int,
    model_f: Any,
    try_times: int = 10,
    train_num: int = 2000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> Dict[str, Any]:
    ds1 ,ds2 = ds
    save_model = None
    for try_idx in range(try_times):
        model = model_f(feature_num)
        random.shuffle(ds1)
        ds10 = ds1[:train_num]
        random.shuffle(ds2)
        ds20 = ds2[:test_num]
        dl, dl2 = generate_dataloader_dataset(
            ds10,  # type:ignore
            ds20,  # type:ignore
            train_num=train_num,
            test_num=test_num,
            batch_size=batch_size,
            batch_size_test=batch_size_test,
        )
        train_model(model, dl, dl2, max_epochs=epochs, gpus=gpus)
        if save_model is None:
            save_model = model
        else:
            for part_idx in range(len(save_model.loss_list)):
                if model.loss_list[part_idx] < save_model.loss_list[part_idx]:
                    save_model.decoder_list[part_idx] = model.decoder_list[part_idx]
                    save_model.loss_list[part_idx] = model.loss_list[part_idx]
    acc = test_acc(save_model, test_loader=dl2)  # type:ignore
    ##
    results = [
        acc,
    ]
    results = torch.Tensor(results)
    d: Dict[str, Any] = {}
    d["mean"] = float(results.mean().data)
    d["max"] = float(results.max().data)  # type:ignore
    d["min"] = float(results.min().data)  # type:ignore
    d["results"] = results
    return d


def try_all_generators_choosebest(
    gr: GeneratorRepository,
    gf: FusionRepository,
    feature_num: int,
    model_f: Any,
    try_times: int = 5,
    train_num: int = 1000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> Dict[str, Any]:
    gf_list = [n for n in gf.get_idx_list() if len(n.split("_")) == feature_num]
    results = [
        get_statis_point_choosebest(
            gr,
            gf,
            fusion_idx,
            feature_num,
            model_f,
            try_times=try_times,
            train_num=train_num,
            test_num=test_num,
            batch_size=batch_size,
            batch_size_test=batch_size_test,
            epochs=epochs,
            gpus=gpus,
        )
        for fusion_idx in gf_list
    ]
    results = [r["mean"] for r in results]
    results = torch.Tensor(results)
    d: Dict[str, Any] = {}
    d["mean"] = float(results.mean().data)
    d["max"] = float(results.max().data)  # type:ignore
    d["min"] = float(results.min().data)  # type:ignore
    d["results"] = results
    return d


def try_all_generators_choosebest_dataset(
    ds: str,
    feature_num: int,
    model_f: Any,
    try_times: int = 5,
    train_num: int = 1000,
    test_num: int = 1000,
    batch_size: int = 100,
    batch_size_test: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> Dict[str, Any]:
    if ds == "MNIST":
        # MINST Data
        dst = mnist_dataset_pair()
    elif ds == "FMNIST":
        # Fashion MNIST
        dst = fmnist_dataset_pair()
    elif ds == 'CIFAR':
        # CIFAR 10
        dst = cifar_dataset_pair()
    results = [
        get_statis_point_choosebest_dataset(
            dst, # type:ignore
            fusion_idx,
            feature_num,
            model_f,
            try_times=try_times,
            train_num=train_num,
            test_num=test_num,
            batch_size=batch_size,
            batch_size_test=batch_size_test,
            epochs=epochs,
            gpus=gpus,
        )
        for fusion_idx in ["TestOneTime"]
    ]
    results = [r["mean"] for r in results]
    results = torch.Tensor(results)
    d: Dict[str, Any] = {}
    d["mean"] = float(results.mean().data)
    d["max"] = float(results.max().data)  # type:ignore
    d["min"] = float(results.min().data)  # type:ignore
    d["results"] = results
    return d


def generate_dataloader_MNIST(
    dataset: Any, sample_num: int, batch_size: int, replacement: bool = False
) -> Any:
    """
    replacement means sampling with replacement
    """
    randomSampler = RandomSampler(dataset, replacement=False, num_samples=sample_num)
    dl = DataLoader(
        dataset, batch_size=batch_size, sampler=randomSampler
    )  # type:ignore
    return dl  # type:ignore


def get_one_point_MNIST(
    trainset: Any,
    testset: Any,
    model: pl.LightningModule,
    train_num: int = 2000,
    test_num: int = 1000,
    batch_size: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> float:
    """
    train a model and return test acc
    """
    dl = generate_dataloader_MNIST(trainset, train_num, batch_size)
    dl2 = generate_dataloader_MNIST(testset, test_num, batch_size)
    train_model(model, dl, dl2, max_epochs=epochs, gpus=gpus)
    acc = test_acc(model, test_loader=dl2)
    return acc


def get_statis_point_MNIST(
    trainset: Any,
    testset: Any,
    feature_num: int,
    model_f: Any,
    try_times: int = 10,
    train_num: int = 2000,
    test_num: int = 1000,
    batch_size: int = 100,
    epochs: int = 20,
    gpus: int = 1,
) -> Dict[str, Any]:
    # model = model_f(feature_num)
    results = [
        get_one_point_MNIST(
            trainset,
            testset,
            model_f(feature_num),
            train_num=train_num,
            test_num=test_num,
            batch_size=batch_size,
            epochs=epochs,
            gpus=gpus,
        )
        for _ in range(try_times)
    ]
    results = torch.Tensor(results)
    d: Dict[str, Any] = {}
    d["mean"] = float(results.mean().data)
    d["max"] = float(results.max().data)  # type:ignore
    d["min"] = float(results.min().data)  # type:ignore
    d["results"] = results
    return d
