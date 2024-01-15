import torch
import torch.nn as nn


class FCBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCBN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x


class FCN(nn.Module):
    def __init__(self, num_layers=3):
        super(FCN, self).__init__()
        self.num_layers = num_layers
        self.module_list = nn.ModuleList()
        self.module_list.append(FCBN(3072, 512))
        for i in range(1, self.num_layers - 1):
            in_channels = 512
            out_channels = 512
            self.module_list.append(FCBN(in_channels, out_channels))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(self.num_layers - 1):
            x = torch.relu(self.module_list[i](x))
        x = self.fc(x)
        return x


class FCN_WOBN(nn.Module):
    def __init__(self, num_layers=3):
        super(FCN_WOBN, self).__init__()
        self.num_layers = num_layers
        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(3072, 512))
        for i in range(1, self.num_layers - 1):
            in_channels = 512
            out_channels = 512
            self.module_list.append(nn.Linear(in_channels, out_channels))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i in range(self.num_layers - 1):
            x = torch.relu(self.module_list[i](x))
        x = self.fc(x)
        return x


# if __name__ == "__main__":
#     input = torch.randn(2, 3, 32, 32)
#     for i in [3, 6, 8, 16, 32, 64]:
#         model = FCN(i)
#         output = model(input)
#         print(output.size())

#         model = FCN_WOBN(i)
#         output = model(input)
#         print(output.size())


# class FC3(nn.Module):
#     def __init__(self):
#         super(FC3, self).__init__()
#         self.fc1 = nn.Linear(3072, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.bn1(self.fc1(x)))
#         x = torch.relu(self.bn2(self.fc2(x)))
#         x = self.fc3(x)
#         return x


# class FC3_WOBN(nn.Module):
#     def __init__(self):
#         super(FC3_WOBN, self).__init__()
#         self.fc1 = nn.Linear(3072, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
