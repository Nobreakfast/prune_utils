import torch
import torch.nn as nn


class CONVBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CONVBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class CONVN(nn.Module):
    def __init__(self, num_layers=3):
        super(CONVN, self).__init__()
        self.num_layers = num_layers
        self.module_list = nn.ModuleList()
        self.module_list.append(CONVBN(3, 8, 3, 1, 1))
        stride = 2
        feature_map = 32
        for i in range(1, self.num_layers):
            in_channels = 8 * 2 ** (i - 1)
            out_channels = 8 * 2**i
            if in_channels >= 512:
                in_channels = 512
                out_channels = 512
            if feature_map == 1:
                stride = 1
            else:
                feature_map = feature_map // 2
            self.module_list.append(CONVBN(in_channels, out_channels, 3, stride, 1))
        self.fc = nn.Linear(out_channels * feature_map * feature_map, 10)

    def forward(self, x):
        for i in range(self.num_layers):
            x = torch.relu(self.module_list[i](x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CONVN_WOBN(nn.Module):
    def __init__(self, num_layers=3):
        super(CONVN_WOBN, self).__init__()
        self.num_layers = num_layers
        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Conv2d(3, 8, 3, 1, 1))
        stride = 2
        feature_map = 32
        for i in range(1, self.num_layers):
            in_channels = 8 * 2 ** (i - 1)
            out_channels = 8 * 2**i
            if in_channels >= 512:
                in_channels = 512
                out_channels = 512
            if feature_map == 1:
                stride = 1
            else:
                feature_map = feature_map // 2
            self.module_list.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        self.fc = nn.Linear(out_channels * feature_map * feature_map, 10)

    def forward(self, x):
        for i in range(self.num_layers):
            x = torch.relu(self.module_list[i](x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# def convN(num_layers):
#     return CONVN(num_layers)


# def convN_WOBN(num_layers):
#     return CONVN_WOBN(num_layers)


# if __name__ == "__main__":
#     x = torch.randn(2, 3, 32, 32)
#     for i in [3, 6, 8, 16, 32, 64]:
#         model = convN(i)
#         y = model(x)
#         print(y.shape)

#         model = convN_WOBN(i)
#         y = model(x)
#         print(y.shape)
