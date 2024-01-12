import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import tqdm
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    class LinearModel(nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, 2, 1)
            self.conv2 = nn.Conv2d(8, 16, 3, 2, 1)
            self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = LinearModel()

    lipz = [
        [
            "module",
            "baseline",
            "baseline_p0.5",
            "xavier",
            "xavier_p0.5",
            "kaiming_out",
            "kaiming_out_p0.5",
            "kaiming_in",
            "kaiming_in_p0.5",
            "haocheng",
            "haocheng_p0.5",
        ]
    ]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            lipz_tmp = [
                name,
            ]
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )
            ifprune = torch.rand(m.weight.shape) > 0.5
            m.weight.data[ifprune] = 0.0
            # m.weight.data *= 1.5
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )

            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )
            ifprune = torch.rand(m.weight.shape) > 0.5
            m.weight.data[ifprune] = 0.0
            # m.weight.data *= 1.5
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )

            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )
            ifprune = torch.rand(m.weight.shape) > 0.5
            m.weight.data[ifprune] = 0.0
            # m.weight.data *= 1.5
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )

            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )
            ifprune = torch.rand(m.weight.shape) > 0.5
            m.weight.data[ifprune] = 0.0
            # m.weight.data *= 1.5
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )

            # haocheng init
            m.weight.data = torch.randn(m.weight.shape)
            m.weight.data /= torch.linalg.norm(
                m.weight.view(m.weight.shape[0], -1), ord=2
            )
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )
            ifprune = torch.rand(m.weight.shape) > 0.5
            m.weight.data[ifprune] = 0.0
            # m.weight.data *= 1.5
            lipz_tmp.append(
                torch.linalg.norm(m.weight.view(m.weight.shape[0], -1).T, ord=2).item()
            )

            lipz.append(lipz_tmp)

    # multiply the column results
    lipz_tmp = [
        "lipschitz",
    ]
    for i in range(1, len(lipz[0])):
        tmp = 1
        for j in range(1, len(lipz)):
            tmp *= lipz[j][i]
        lipz_tmp.append(tmp)

    lipz.append(lipz_tmp)
    with open("logs/lipz_conv.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(lipz)
