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
        "--num_samples", type=int, default=10000, help="number of samples"
    )
    args = parser.parse_args()

    # 生成正态分布的随机数据
    data = np.random.normal(args.mu, args.sigma, args.num_samples)

    # 绘制直方图
    plt.hist(data, bins=50, density=True, alpha=0.6, color="g")

    # 添加标签和标题
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal Distribution")

    # 添加正态分布的密度曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-((x - args.mu) ** 2) / (2 * args.sigma**2)) / (
        args.sigma * np.sqrt(2 * np.pi)
    )
    plt.plot(x, p, "k", linewidth=2)
    plt.xlim(-0.55, 0.55)
    plt.ylim(0, 10)
    # plt.axhline(0, color="black", lw=2)
    plt.axvline(0, color="black", lw=2)
    plt.tight_layout()

    # 显示图形
    plt.show()
