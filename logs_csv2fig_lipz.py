import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=str, default="results/basic_lipz/lipz_raw.csv"
    )
    parser.add_argument("-d", "--dest", type=str, default="./results/basic_lipz")
    args = parser.parse_args()

    # os.system("mkdir -p " + args.dest)

    # data_raw = pd.read_csv(args.file)

    # data = []
    # for name, group in data_raw.groupby(
    #     [
    #         "data",
    #         "m",
    #         "im",
    #         "pa",
    #         "pm",
    #         "res",
    #         "layer",
    #     ]
    # ):
    #     data.append(
    #         [
    #             name[0],
    #             name[1],
    #             name[2],
    #             name[3],
    #             name[4],
    #             name[5],
    #             name[6],
    #             group["ilipz"].mean(),
    #             group["plipz"].mean(),
    #             group["rlipz"].mean(),
    #             group["ivar"].mean(),
    #             group["pvar"].mean(),
    #             group["rvar"].mean(),
    #             group["imean"].mean(),
    #             group["pmean"].mean(),
    #             group["rmean"].mean(),
    #         ]
    #     )

    # data = pd.DataFrame(
    #     data,
    #     columns=[
    #         "data",
    #         "m",
    #         "im",
    #         "pa",
    #         "pm",
    #         "res",
    #         "layer",
    #         "ilipz",
    #         "plipz",
    #         "rlipz",
    #         "ivar",
    #         "pvar",
    #         "rvar",
    #         "imean",
    #         "pmean",
    #         "rmean",
    #     ],
    # )
    # data.to_csv(args.dest + "/output_lipz.csv", index=False)
    data = pd.read_csv(args.dest + "/output_lipz.csv")

    for model in data["m"].unique():
        for init_method in data["im"].unique():
            tmp_data = data[
                (data["m"] == model)
                & (data["im"] == init_method)
                # & (data["pa"] == algo)
                # & (data["layer"] == layer)
                & (data["res"] == 1)
            ]
            for layer in tmp_data["layer"].unique():
                tmp_data = data[
                    (data["m"] == model)
                    & (data["im"] == init_method)
                    # & (data["pa"] == algo)
                    # & (data["layer"] == layer)
                    & (data["res"] == 1)
                ]
                tmp_data = tmp_data.sort_values(by=["pm"])

                tmp_data = tmp_data[(tmp_data["layer"] == layer)]

                colors = ["b", "g", "r", "c", "m", "y", "k"]
                algo_list = ["uniform", "snip", "synflow"]
                # plt size
                plt.figure()
                plt.subplot(3, 1, 1)
                count = 0
                for algo in algo_list:
                    plt.plot(
                        tmp_data[tmp_data["pa"] == algo]["pm"],
                        tmp_data[tmp_data["pa"] == algo]["imean"],
                        # label=algo + "_init",
                        linestyle="-",
                        color=colors[count],
                    )
                    plt.plot(
                        tmp_data[tmp_data["pa"] == algo]["pm"],
                        tmp_data[tmp_data["pa"] == algo]["pmean"],
                        label=algo,
                        linestyle="--",
                        color=colors[count],
                    )
                    count += 1
                # remove x axis value, only save the tick
                plt.gca().axes.get_xaxis().set_ticklabels([])
                plt.ylabel("mean")
                plt.grid(True)

                plt.subplot(3, 1, 2)
                count = 0
                for algo in algo_list:
                    plt.plot(
                        tmp_data[tmp_data["pa"] == algo]["pm"],
                        tmp_data[tmp_data["pa"] == algo]["ilipz"],
                        # label=algo + "_init",
                        linestyle="-",
                        color=colors[count],
                    )
                    plt.plot(
                        tmp_data[tmp_data["pa"] == algo]["pm"],
                        tmp_data[tmp_data["pa"] == algo]["plipz"],
                        label=algo,
                        linestyle="--",
                        color=colors[count],
                    )
                    count += 1
                plt.legend()
                plt.gca().axes.get_xaxis().set_ticklabels([])
                plt.ylabel("lipz")
                plt.grid(True)

                plt.subplot(3, 1, 3)
                count = 0
                for algo in algo_list:
                    plt.plot(
                        tmp_data[tmp_data["pa"] == algo]["pm"],
                        tmp_data[tmp_data["pa"] == algo]["ivar"],
                        # label=algo + "_init",
                        linestyle="-",
                        color=colors[count],
                    )
                    plt.plot(
                        tmp_data[tmp_data["pa"] == algo]["pm"],
                        tmp_data[tmp_data["pa"] == algo]["pvar"],
                        label=algo,
                        linestyle="--",
                        color=colors[count],
                    )
                    count += 1
                # plt.legend()
                plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                plt.xlabel("Prune Ratio")
                plt.ylabel("var")
                plt.grid(True)

                plt.savefig(
                    args.dest + "/" + model + "_" + init_method + "_" + layer + ".png"
                )
                # plt.savefig(args.dest + "/test.png")
                plt.close()
