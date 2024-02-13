import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="lipz_raw.csv")
    parser.add_argument("-d", "--dest", type=str, default="./results/basic")
    args = parser.parse_args()

    os.system("mkdir -p " + args.dest)

    data_raw = pd.read_csv(args.file)

    data = []
    for name, group in data_raw.groupby(
        [
            "data",
            "m",
            "im",
            "pa",
            "pm",
            "res",
            "layer",
        ]
    ):
        data.append(
            [
                name[0],
                name[1],
                name[2],
                name[3],
                name[4],
                name[5],
                name[6],
                group["ilipz"].mean(),
                group["plipz"].mean(),
                group["rlipz"].mean(),
                group["ivar"].mean(),
                group["pvar"].mean(),
                group["rvar"].mean(),
                group["imean"].mean(),
                group["pmean"].mean(),
                group["rmean"].mean(),
            ]
        )

    data = pd.DataFrame(
        data,
        columns=[
            "data",
            "m",
            "im",
            "pa",
            "pm",
            "res",
            "layer",
            "ilipz",
            "plipz",
            "rlipz",
            "ivar",
            "pvar",
            "rvar",
            "imean",
            "pmean",
            "rmean",
        ],
    )
    data.to_csv(args.dest + "/output_lipz.csv", index=False)

    model_list = data["model"].unique()
    init_list = data["init method"].unique()
    marker_list = [
        "o",
        "v",
        "^",
        "<",
        ">",
        "s",
        "p",
        "P",
        "*",
        "h",
        "H",
        "+",
        "x",
        "X",
        "D",
    ][0 : len(init_list)]

    for model in model_list:
        model_data = data[data["model"] == model]
        model_data = model_data.sort_values(by=["prune ratio"])
        model_data = model_data.reset_index(drop=True)

        plt.figure()
        for init_method, marker in zip(init_list, marker_list):
            tmp_data = model_data[model_data["init method"] == init_method]
            tmp_data = tmp_data[tmp_data["restore"] == 0]
            plt.plot(
                tmp_data["prune ratio"] * 100,
                tmp_data["mean"],
                color="red",
                label=init_method,
                marker=marker,
            )
            plt.fill_between(
                tmp_data["prune ratio"] * 100,
                tmp_data["min"],
                tmp_data["max"],
                color="red",
                alpha=0.2,
                # label="min-max",
            )

            tmp_data = model_data[model_data["init method"] == init_method]
            tmp_data = tmp_data[tmp_data["restore"] == 1]
            tmp_data["prune ratio"] = tmp_data["prune ratio"] + 0.005

            plt.plot(
                tmp_data["prune ratio"] * 100,
                tmp_data["mean"],
                color="blue",
                label=init_method + "_r1",
                linestyle="--",
                marker=marker,
            )
            plt.fill_between(
                tmp_data["prune ratio"] * 100,
                tmp_data["min"],
                tmp_data["max"],
                color="blue",
                alpha=0.2,
                # label="min-max",
            )

            # restore 2
            tmp_data = model_data[model_data["init method"] == init_method]
            tmp_data = tmp_data[tmp_data["restore"] == 2]
            tmp_data["prune ratio"] = tmp_data["prune ratio"] + 0.010

            plt.plot(
                tmp_data["prune ratio"] * 100,
                tmp_data["mean"],
                color="green",
                label=init_method + "_r2",
                linestyle="--",
                marker=marker,
            )
            plt.fill_between(
                tmp_data["prune ratio"] * 100,
                tmp_data["min"],
                tmp_data["max"],
                color="green",
                alpha=0.2,
                # label="min-max",
            )

            # restore 3
            tmp_data = model_data[model_data["init method"] == init_method]
            tmp_data = tmp_data[tmp_data["restore"] == 3]
            tmp_data["prune ratio"] = tmp_data["prune ratio"] + 0.015

            plt.plot(
                tmp_data["prune ratio"] * 100,
                tmp_data["mean"],
                color="purple",
                label=init_method + "_r3",
                linestyle="--",
                marker=marker,
            )
            plt.fill_between(
                tmp_data["prune ratio"] * 100,
                tmp_data["min"],
                tmp_data["max"],
                color="purple",
                alpha=0.2,
                # label="min-max",
            )

        # plt.yscale("log")
        plt.xticks(range(80, 100, 2))
        plt.xlabel("Prune Ratio")
        plt.ylabel("accuracy")
        plt.legend()
        # line of graph
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(args.dest + "/" + model + ".png")
