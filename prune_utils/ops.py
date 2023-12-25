import torch
import torch.nn as nn


def get_lipz_fc(weight):
    """
    Compute the Lipschitz bound of a fully connected layer
    """
    # calculate the spectral norm of the weight matrix
    u, s, v = torch.svd(weight)
    # print(torch.linalg.norm(weight, ord=2))
    return s[0]


def get_lipz_conv(weight):
    """
    Compute the Lipschitz bound of a convolutional layer
    """
    # calculate the spectral norm of the weight matrix
    u, s, v = torch.svd(weight.view(weight.shape[0], -1))
    return s[0]


def get_lipz_conv_pi(weight, iter=50):
    """
    Compute the Lipschitz bound of a convolutional layer
    """
    # random initialize a x0
    w = weight.view(weight.shape[0], -1)
    x0 = torch.randn(w.shape[1], 1)
    x0 = x0 / torch.norm(x0)
    wmatmul = w.t().matmul(w)
    # power iteration
    for i in range(iter):
        x0 = wmatmul.matmul(x0)
        x0 = x0 / torch.norm(x0)
    # calculate the spectral norm of the weight matrix
    return torch.norm(w.matmul(x0))


if __name__ == "__main__":
    weight = torch.randn(64, 64)
    print(get_lipz_fc(weight))
    weight = torch.randn(64, 64, 3, 3)
    print(get_lipz_conv(weight))
    weight = torch.randn(64, 64, 3, 3)
    print(get_lipz_conv_pi(weight))
