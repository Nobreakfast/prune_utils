import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, default="results_raw.csv")
    parser.add_argument("-d", "--dest", type=str, default="./results/basic")
    args = parser.parse_args()

    os.system("mkdir -p " + args.dest)

    data_raw = pd.read_csv(args.file)

    data = []
    for name, group in data_raw.groupby(
        ["data type", "model", "init method", "prune ratio", "restore"]
    ):
        data.append(
            [
                name[0],
                name[1],
                name[2],
                name[3],
                name[4],
                group["best accuracy"].mean(),
                group["best accuracy"].max(),
                group["best accuracy"].min(),
                group["best accuracy"].std(),
            ]
        )

    data = pd.DataFrame(
        data,
        columns=[
            "data type",
            "model",
            "init method",
            "prune ratio",
            "restore",
            "mean",
            "max",
            "min",
            "std",
        ],
    )
    data.to_csv(args.dest + "/output.csv", index=False)

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

    plt.figure()
    x = ["fc3_ki", "fc3_ko", "fc3_xa", "fc3_wobn_ki", "fc3_wobn_ko", "fc3_wobn_xa"]
    bl, bl_min, bl_max = [], [], []
    r1, r1_min, r1_max = [], [], []
    r2, r2_min, r2_max = [], [], []
    r3, r3_min, r3_max = [], [], []
    for model in ["fc3", "fc3_wobn"]:
        model_data = data[data["model"] == model]
        for im in ["kaiming_in", "kaiming_out", "xavier"]:
            tmp_data = model_data[model_data["init method"] == im]
            tmp_data = tmp_data[tmp_data["restore"] == 0]
            bl.append(tmp_data["mean"].values[0])
            bl_min.append(tmp_data["min"].values[0])
            bl_max.append(tmp_data["max"].values[0])
            tmp_data = model_data[model_data["init method"] == im]
            tmp_data = tmp_data[tmp_data["restore"] == 1]
            r1.append(tmp_data["mean"].values[0])
            r1_min.append(tmp_data["min"].values[0])
            r1_max.append(tmp_data["max"].values[0])
            try:
                tmp_data = model_data[model_data["init method"] == im]
                tmp_data = tmp_data[tmp_data["restore"] == 2]
                r2.append(tmp_data["mean"].values[0])
                r2_min.append(tmp_data["min"].values[0])
                r2_max.append(tmp_data["max"].values[0])
                tmp_data = model_data[model_data["init method"] == im]
                tmp_data = tmp_data[tmp_data["restore"] == 3]
                r3.append(tmp_data["mean"].values[0])
                r3_min.append(tmp_data["min"].values[0])
                r3_max.append(tmp_data["max"].values[0])
            except:
                r2.append(0)
                r2_min.append(0)
                r2_max.append(0)
                r3.append(0)
                r3_min.append(0)
                r3_max.append(0)
    x = np.arange(len(x))
    plt.scatter(x, bl, label="baseline", marker="o", color="red")
    plt.scatter(x, bl_max, marker="v", color="red")
    plt.scatter(x, bl_min, marker="^", color="red")
    plt.plot([x, x], [bl_min, bl_max], color="red")
    x = x + 0.15
    plt.scatter(x, r1, label="restore weight", marker="o", color="blue")
    plt.scatter(x, r1_max, marker="v", color="blue")
    plt.scatter(x, r1_min, marker="^", color="blue")
    plt.plot([x, x], [r1_min, r1_max], color="blue")
    x = x + 0.15
    plt.scatter(x, r2, label="restore BN", marker="o", color="green")
    plt.scatter(x, r2_max, marker="v", color="green")
    plt.scatter(x, r2_min, marker="^", color="green")
    plt.plot([x, x], [r2_min, r2_max], color="green")
    x = x + 0.15
    plt.scatter(x, r3, label="restore weight+BN", marker="o", color="purple")
    plt.scatter(x, r3_max, marker="v", color="purple")
    plt.scatter(x, r3_min, marker="^", color="purple")
    plt.plot([x, x], [r3_min, r3_max], color="purple")
    plt.xlabel("init method and model")
    plt.ylabel("accuracy")
    plt.xticks(
        np.arange(len(x)),
        ["fc3_ki", "fc3_ko", "fc3_xa", "fc3_wobn_ki", "fc3_wobn_ko", "fc3_wobn_xa"],
    )
    plt.ylim(50, 70)
    plt.legend()
    plt.xticks(rotation=30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(args.dest + "/fc3_wobn.png")

    plt.figure()
    x = [
        "conv3_ki",
        "conv3_ko",
        "conv3_xa",
        "conv3_wobn_ki",
        "conv3_wobn_ko",
        "conv3_wobn_xa",
    ]
    bl, bl_min, bl_max = [], [], []
    r1, r1_min, r1_max = [], [], []
    r2, r2_min, r2_max = [], [], []
    r3, r3_min, r3_max = [], [], []
    for model in ["conv3", "conv3_wobn"]:
        model_data = data[data["model"] == model]
        for im in ["kaiming_in", "kaiming_out", "xavier"]:
            tmp_data = model_data[model_data["init method"] == im]
            tmp_data = tmp_data[tmp_data["restore"] == 0]
            bl.append(tmp_data["mean"].values[0])
            bl_min.append(tmp_data["min"].values[0])
            bl_max.append(tmp_data["max"].values[0])
            tmp_data = model_data[model_data["init method"] == im]
            tmp_data = tmp_data[tmp_data["restore"] == 1]
            r1.append(tmp_data["mean"].values[0])
            r1_min.append(tmp_data["min"].values[0])
            r1_max.append(tmp_data["max"].values[0])
            try:
                tmp_data = model_data[model_data["init method"] == im]
                tmp_data = tmp_data[tmp_data["restore"] == 2]
                r2.append(tmp_data["mean"].values[0])
                r2_min.append(tmp_data["min"].values[0])
                r2_max.append(tmp_data["max"].values[0])
                tmp_data = model_data[model_data["init method"] == im]
                tmp_data = tmp_data[tmp_data["restore"] == 3]
                r3.append(tmp_data["mean"].values[0])
                r3_min.append(tmp_data["min"].values[0])
                r3_max.append(tmp_data["max"].values[0])
            except:
                r2.append(0)
                r2_min.append(0)
                r2_max.append(0)
                r3.append(0)
                r3_min.append(0)
                r3_max.append(0)
    x = np.arange(len(x))
    plt.scatter(x, bl, label="baseline", marker="o", color="red")
    plt.scatter(x, bl_max, marker="v", color="red")
    plt.scatter(x, bl_min, marker="^", color="red")
    plt.plot([x, x], [bl_min, bl_max], color="red")
    x = x + 0.15
    plt.scatter(x, r1, label="restore weight", marker="o", color="blue")
    plt.scatter(x, r1_max, marker="v", color="blue")
    plt.scatter(x, r1_min, marker="^", color="blue")
    plt.plot([x, x], [r1_min, r1_max], color="blue")
    x = x + 0.15
    plt.scatter(x, r2, label="restore BN", marker="o", color="green")
    plt.scatter(x, r2_max, marker="v", color="green")
    plt.scatter(x, r2_min, marker="^", color="green")
    plt.plot([x, x], [r2_min, r2_max], color="green")
    x = x + 0.15
    plt.scatter(x, r3, label="restore weight+BN", marker="o", color="purple")
    plt.scatter(x, r3_max, marker="v", color="purple")
    plt.scatter(x, r3_min, marker="^", color="purple")
    plt.plot([x, x], [r3_min, r3_max], color="purple")

    plt.xlabel("init method and model")
    plt.ylabel("accuracy")
    plt.xticks(
        np.arange(len(x)),
        [
            "conv3_ki",
            "conv3_ko",
            "conv3_xa",
            "conv3_wobn_ki",
            "conv3_wobn_ko",
            "conv3_wobn_xa",
        ],
    )
    plt.ylim(60, 85)
    plt.legend()
    plt.xticks(rotation=30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(args.dest + "/conv3_wobn.png")
