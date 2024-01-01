import torch
import torch.nn as nn
import numpy as np
import scipy


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


def objective_function(var, spec_norm, W):
    return spec_norm**2 / (0.5 * W * var) - 1


def binary_search(
    H, W, lower_bound=0, upper_bound=0.5, tolerance=1e-4, max_iterations=500
):
    for _ in range(max_iterations):
        mid_point = (lower_bound + upper_bound) / 2.0
        matrix = np.random.normal(0, mid_point, size=(H, W))
        spec_norm = np.linalg.norm(matrix, ord=2)
        f_value = objective_function(mid_point, spec_norm, W)

        if abs(f_value) < tolerance:
            return mid_point, matrix

        if f_value > 0:
            upper_bound = mid_point
        else:
            lower_bound = mid_point

    return (lower_bound + upper_bound) / 2.0, matrix


if __name__ == "__main__":
    weight = torch.randn(64, 64)
    print(get_lipz_fc(weight))
    weight = torch.randn(64, 64, 3, 3)
    print(get_lipz_conv(weight))
    weight = torch.randn(64, 64, 3, 3)
    print(get_lipz_conv_pi(weight))
