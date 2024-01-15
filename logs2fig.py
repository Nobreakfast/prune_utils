import torch
import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse


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

    # 遍历logs文件夹中的每个子文件夹
    for subdir in os.listdir(logs_folder):
        subdir_path = os.path.join(logs_folder, subdir)

        # 检查是否是文件夹
        if os.path.isdir(subdir_path):
            # 获取每个子文件夹中的最佳accuracy
            best_accuracy = get_best_accuracy(subdir_path)
            results[subdir] = best_accuracy

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tensorboard log parser")
    parser.add_argument("-p", "--path", help="logs folder path", default="logs")
    args = parser.parse_args()

    # 处理logs文件夹
    results = process_logs_folder(args.path)

    # 打印结果
    for subdir, best_accuracy in results.items():
        print(f"Folder: {subdir}, Best Test Accuracy: {best_accuracy}")
