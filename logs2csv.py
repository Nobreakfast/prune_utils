import torch
import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import pandas as pd


def get_best_accuracy(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # 获取所有的summary keys
    tags = event_acc.Tags()["scalars"]

    best_accuracy = 0.0

    for tag in tags:
        # 在每个tag下找到最大的test accuracy
        if "test accuracy" in tag:
            events = event_acc.Scalars(tag)
            accuracies = [event.value for event in events]
            max_accuracy = max(accuracies)

            # 更新最佳accuracy
            if max_accuracy > best_accuracy:
                best_accuracy = max_accuracy

    return best_accuracy


def process_logs_folder(logs_folder):
    results = {}
    if os.path.isdir(logs_folder):
        best_accuracy = get_best_accuracy(logs_folder)
        if best_accuracy != 0.0:
            results[logs_folder] = best_accuracy
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
    for subdir, best_accuracy in results.items():
        subdir = subdir.split("/")
        data_type = subdir[1]
        model = subdir[2]
        filename = subdir[3]
        filename = filename.split("_")
        number = filename[-1][-1]
        restore = filename[-2][1]
        prune_ratio = filename[-3][1:]
        init_method = "_".join(filename[:-3])
        # print(data_type, model, init_method, prune_ratio, restore, number)
        data_raw.append(
            [
                data_type,
                model,
                init_method,
                prune_ratio,
                restore,
                number,
                best_accuracy,
            ]
        )

    data_raw = pd.DataFrame(
        data_raw,
        columns=[
            "data type",
            "model",
            "init method",
            "prune ratio",
            "restore",
            "number",
            "best accuracy",
        ],
    )

    data_raw.to_csv(args.dest + "/results_raw.csv", index=False)
