import torch
import torch.nn as nn


@torch.no_grad()
def cal_lipz(
    model: nn.Module, x_shape, eps=1e-7, norm=2, times=1000, device=torch.device("cpu")
):
    """
    Compute the Lipschitz bound of a model on a batch of inputs
    """
    model.to(device)
    if len(x_shape) == 4:
        dim_x = tuple(range(len(x_shape))[1:])
        tmp_out = model(torch.randn(x_shape).to(device))
        dim_y = tuple(range(len(tmp_out.shape))[1:])
    elif len(x_shape) == 3:
        raise NotImplementedError
    elif len(x_shape) == 2:
        raise NotImplementedError
    else:
        raise ValueError("Input shape should be 2, 3 or 4")
    max_lipz = 0
    for i in trange(times):
        x1 = torch.randn(x_shape)
        x2 = x1 + eps * torch.sign(torch.randn(x_shape))
        # x2 = x1 + eps * torch.randn(x_shape)
        x1 = x1.to(device)
        x2 = x2.to(device)
        y1 = model(x1)
        y2 = model(x2)
        y_norm = torch.norm(y1 - y2, dim=dim_y, p=norm)
        x_norm = torch.norm(x1 - x2, dim=dim_x, p=norm)
        lipz = torch.max(y_norm / x_norm)
        if lipz > max_lipz:
            max_lipz = lipz
    return max_lipz


if __name__ == "__main__":
    from tqdm import trange
    from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

    layer_num = 7
    model = ResNet(BasicBlock, [2] * layer_num)
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # global random prune on all convolution module
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data *= torch.rand_like(m.weight.data) > 0.5

    model.eval()
    print(cal_lipz(model, (2, 3, 32, 32)))
