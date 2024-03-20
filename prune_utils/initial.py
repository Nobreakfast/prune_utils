import torch
import torch.nn as nn
import math


def genOrthgonal(dim):
    """
    github repo: https://github.com/JiJingYu/delta_orthogonal_init_pytorch
    MIT License

    Copyright (c) 2018 JingYu Ji

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.linalg.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    """
    github repo: https://github.com/JiJingYu/delta_orthogonal_init_pytorch
    MIT License

    Copyright (c) 2018 JingYu Ji

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    weights.data[:, :, mid1, mid2] = q[: weights.size(0), : weights.size(1)]
    weights.data.mul_(gain)


def makeOrthogonal(weights, gain):
    """
    github repo: https://github.com/iassael/torch-linearo/blob/master/LinearO.lua
    Copyright (C) 2015 John-Alexander M. Assael (www.johnassael.com)
    Copyright (C) 2024 Nobreakfast

    The MIT License (MIT)

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    initScale = math.sqrt(2)
    M1 = torch.randn(weights.size(0), weights.size(0))
    M2 = torch.randn(weights.size(1), weights.size(1))
    n_min = min(weights.size(0), weights.size(1))
    Q1, R1 = torch.linalg.qr(M1)
    Q2, R2 = torch.linalg.qr(M2)
    # weights.data.copy_(Q1.narrow(1, 0, n_min) * Q2.narrow(0, 0, n_min)).mul_(initScale)
    weights.data.copy_(torch.mm(Q1.narrow(1, 0, n_min), Q2.narrow(0, 0, n_min))).mul_(
        initScale
    )


def initialization(model, method, **kwargs):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if method == "xavier":
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            elif method == "kaiming_out":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif method == "kaiming_in":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif method == "orthogonal":
                makeDeltaOrthogonal(m.weight, gain=nn.init.calculate_gain("relu"))
            else:
                pass
        elif isinstance(m, nn.Linear):
            if method == "xavier":
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            elif method == "kaiming_out":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif method == "kaiming_in":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif method == "orthogonal":
                makeOrthogonal(m.weight, gain=nn.init.calculate_gain("relu"))
            else:
                pass


if __name__ == "__main__":
    conv = nn.Conv2d(3, 6, 3, 1, 1)
    print(conv.weight.data)
    makeDeltaOrthogonal(conv.weight, gain=nn.init.calculate_gain("relu"))
    print(conv.weight.data)
    fc = nn.Linear(4, 8)
    print(fc.weight.data)
    makeOrthogonal(fc.weight, gain=nn.init.calculate_gain("relu"))
    print(fc.weight.data)
