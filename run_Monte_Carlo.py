import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os
from CodeLib import GeneratorRepository, FusionRepository
from CodeLib import try_all_generators, try_all_generators_choosebest
from CodeLib import generate_allinone_model, generate_seperate_model, generate_allinone_vgg16,generate_allinone_res
from CodeLib import make_sure_fold_exists

if __name__ == '__main__':
    # load generators
    gr = GeneratorRepository("./model/generator/")
    gf = FusionRepository("./model/fusion/")

    # setting
    try_times = 4

    # try list
    train_num_list = [(i + 1) * 10 for i in range(10)] + [(i+1)*100 for i in range(1,10)]
    batch_size_list = [10 for _ in range(10)] + [10 for _ in range(1,10)]
    batch_size_list_test = [100 for _ in range(10)] + [100 for _ in range(1,10)]

    # save path
    save_path = "./output"
    make_sure_fold_exists(save_path)

    all_in_one_vgg = True
    all_in_one_res = True
    seperate_model = True
    seperate_model_choose_best = True

    # all in one vgg
    if all_in_one_vgg:
        overall_d = [
            try_all_generators(
                gr,
                gf,
                4,
                generate_allinone_vgg16,
                try_times=try_times,
                train_num=train_num_list[i],
                test_num=100,
                batch_size=batch_size_list[i],
                batch_size_test = batch_size_list_test[i],
            )
            for i in range(len(train_num_list))
        ]

        # save
        save_path1 = os.path.join(save_path, "all_in_one_vgg_monte_carlo.csv")
        with open(save_path1, "w") as f:
            for idx, mean_value, max_value in zip(
                train_num_list,
                [d["mean"] for d in overall_d],
                [d["max"] for d in overall_d],
            ):
                f.write(str(idx) + "," + str(mean_value) + "," + str(max_value) + "," + "\n")

    # all in one res
    if all_in_one_res:
        overall_d = [
            try_all_generators(
                gr,
                gf,
                4,
                generate_allinone_res,
                try_times=try_times,
                train_num=train_num_list[i],
                test_num=100,
                batch_size=batch_size_list[i],
                batch_size_test = batch_size_list_test[i],
            )
            for i in range(len(train_num_list))
        ]

        # save
        save_path1 = os.path.join(save_path, "all_in_one_res_monte_carlo.csv")
        with open(save_path1, "w") as f:
            for idx, mean_value, max_value in zip(
                train_num_list,
                [d["mean"] for d in overall_d],
                [d["max"] for d in overall_d],
            ):
                f.write(str(idx) + "," + str(mean_value) + "," + str(max_value) + "," + "\n")

    # seperate model
    if seperate_model:
        overall_d = [
            try_all_generators(
                gr,
                gf,
                4,
                generate_seperate_model,
                try_times=try_times,
                train_num=train_num_list[i],
                test_num=100,
                batch_size=batch_size_list[i],
                batch_size_test = batch_size_list_test[i],
            )
            for i in range(len(train_num_list))
        ]
        # overall_d = [try_all_generators(gr, gf,4,generate_seperate_model,try_times=1,train_num=(i+1)*10,test_num=100,batch_size=10) for i in range(1)]
        # save
        save_path2 = os.path.join(save_path, "seperate_monte_carlo.csv")
        with open(save_path2, "w") as f:
            for idx, mean_value, max_value in zip(
                train_num_list,
                [d["mean"] for d in overall_d],
                [d["max"] for d in overall_d],
            ):
                f.write(str(idx) + "," + str(mean_value) + "," + str(max_value) + "," + "\n")

    # seperate model choose best
    if seperate_model_choose_best:
        overall_d = [
            try_all_generators_choosebest(
                gr,
                gf,
                4,
                generate_seperate_model,
                try_times=try_times,
                train_num=train_num_list[i],
                test_num=100,
                batch_size=batch_size_list[i],
                batch_size_test = batch_size_list_test[i],
            )
            for i in range(len(train_num_list))
        ]
        # overall_d = [try_all_generators_choosebest(gr, gf,4,generate_seperate_model,try_times=3,train_num=(i+1)*10,test_num=100,batch_size=10) for i in range(1)]
        # save
        save_path2 = os.path.join(save_path, "seperate_best_monte_carlo.csv")
        with open(save_path2, "w") as f:
            for idx, mean_value, max_value in zip(
                train_num_list,
                [d["mean"] for d in overall_d],
                [d["max"] for d in overall_d],
            ):
                f.write(str(idx) + "," + str(mean_value) + "," + str(max_value) + "," + "\n")
