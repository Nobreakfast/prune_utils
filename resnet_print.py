import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import prune
from torch.utils.tensorboard import SummaryWriter
import argparse
import tqdm
import models.resnet as resnet
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "-l", "--load", help="load path", default="./logs/test/best.pth"
    )
    args = parser.parse_args()

    model = resnet.resnet20()
    model.load_state_dict(torch.load(args.load))

    data = pd.DataFrame(
        columns=["name", "in_shape", "out_shape", "sn", "var", "std", "sv", "sv_n"]
    )
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_shape = (
                module.weight.data.shape[1]
                * module.weight.data.shape[2]
                * module.weight.data.shape[3]
            )
            out_shape = module.weight.data.shape[0]
            sn = torch.linalg.norm(
                module.weight.data.view(module.weight.data.size(0), -1), ord=2
            ).item()
            var = module.weight.data.var().item()
            std = module.weight.data.std().item()
            sv = sn / std
            sv_n = sn / std / ((in_shape / 2) ** 0.5)
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "name": name,
                            "in_shape": in_shape,
                            "out_shape": out_shape,
                            "sn": sn,
                            "var": var,
                            "std": std,
                            "sv": sv,
                            "sv_n": sv_n,
                        },
                        index=[0],
                    ),
                ]
            )

        elif isinstance(module, nn.Linear):
            in_shape = module.weight.data.shape[1]
            out_shape = module.weight.data.shape[0]
            sn = torch.linalg.norm(
                module.weight.data.view(module.weight.data.size(0), -1), ord=2
            ).item()
            var = module.weight.data.var().item()
            std = module.weight.data.std().item()
            sv = sn / std
            sv_n = sn / std / ((in_shape / 2) ** 0.5)
            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        {
                            "name": name,
                            "in_shape": in_shape,
                            "out_shape": out_shape,
                            "sn": sn,
                            "var": var,
                            "std": std,
                            "sv": sv,
                            "sv_n": sv_n,
                        },
                        index=[0],
                    ),
                ]
            )
    # change path from args.load "*.pth" to "*.csv"
    path = args.load.replace(".pth", ".csv")
    data.to_csv(path)

    # print data to terminal
    print(data)
