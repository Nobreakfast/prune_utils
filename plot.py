import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a histogram of normal distribution."
    )
    parser.add_argument(
        "--mu", type=float, default=0, help="mean of normal distribution"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="standard deviation of normal distribution",
    )
    parser.add_argument(
        "--num_samples", type=int, default=200, help="number of samples"
    )
    parser.add_argument("--prune", help="prune rate", type=float, default=0.0)
    parser.add_argument("--restore", help="restore type", type=int, default=0)
    args = parser.parse_args()
    # init seed
    np.random.seed(0)

    # 生成正态分布的随机数据
    data = np.random.normal(args.mu, args.sigma, args.num_samples)

    if args.prune > 0:
        numtoprune = int(args.num_samples * args.prune)
        # random zero out
        data[:numtoprune] = 0

    if args.restore > 0:
        # restore
        mean = data[data != 0].mean()
        var = data.var()
        data[data != 0] -= mean
        sn = np.linalg.norm(data, ord=2)
        data[data != 0] /= sn

    # 绘制直方图
    plt.figure()
    plt.hist(data, bins=10, density=True, edgecolor="white")

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=17)
    plt.tight_layout()

    plt.savefig(
        f"results2/hist/hist_r{args.restore}_p{args.prune}_mu{args.mu}_sigma{args.sigma}.png",
        transparent=True,
    )
    # set the font of x, y tickets
    # 显示图形
    # plt.show()
