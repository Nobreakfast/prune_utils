import torch
import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import pandas as pd


def get_data(log_dir):
    initialized = np.fromfile(
        os.path.join(log_dir, "initialized.csv"), dtype=np.float64
    )
    pruned = np.fromfile(os.path.join(log_dir, "pruned.csv"), dtype=np.float64)
    restored = np.fromfile(os.path.join(log_dir, "restored.csv"), dtype=np.float64)
    data = np.concatenate((initialized, pruned[:, 1:], restored[:, 1:]))
    return data


def process_logs_folder(logs_folder):
    results = {}
    if os.path.isdir(logs_folder):
        data = get_data(logs_folder)
        results[logs_folder] = data
        for subdir in os.listdir(logs_folder):
            subdir_path = os.path.join(logs_folder, subdir)
            results.update(process_logs_folder(subdir_path))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tensorboard log parser")
    parser.add_argument("-p", "--path", help="logs folder path", default="logs")
    parser.add_argument("-d", "--dest", help="csv file path", default="./results")
    args = parser.parse_args()

    os.system("mkdir -p " + args.dest)

    results = process_logs_folder(args.path)

    # for subdir, best_accuracy in results.items():
    #     print(f"Folder: {subdir}, Best Test Accuracy: {best_accuracy}")

    data_raw = []
    for subdir, data in results.items():
        subdir = subdir.split("/")
        data_type = subdir[1]
        model_init = subdir[2].split("_")
        model = model_init[0]
        im = model_init[1]
        prune_info = subdir[3].split("_")
        pa = prune_info[0][1:]
        pm = prune_info[1]
        restore = subdir[4][1:]
        number = subdir[5][3:]

        for i in data.shape[0]:
            # print(data_type, model, init_method, prune_ratio, restore, number)
            tmp_data = data[i]
            data_raw.append(
                [
                    data_type,
                    model,
                    im,
                    pa,
                    pm,
                    restore,
                    number,
                    tmp_data[0],
                    tmp_data[1],
                    tmp_data[4],
                    tmp_data[7],
                    tmp_data[2],
                    tmp_data[5],
                    tmp_data[8],
                    tmp_data[3],
                    tmp_data[6],
                    tmp_data[9],
                ]
            )

    data_raw = pd.DataFrame(
        data_raw,
        columns=[
            "data",
            "m",
            "im",
            "pa",
            "pm",
            "res",
            "no.",
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

    data_raw.to_csv(args.dest + "/lipz_raw.csv", index=False)
