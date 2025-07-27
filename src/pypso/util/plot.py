import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_curve(fitness_values: list[float]) -> None:
    """
    绘制适应度函数曲线和对应的对数函数曲线

    Args:
        fitness_values (list[float]): 适应度值，浮点数，一维数组

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.title("Fitness Function Curve")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fitness Value")
    plt.plot(fitness_values, "b-", linewidth=2)
    plt.grid(True)

    plt.subplot(122)
    plt.title("Fitness Function Curve (log)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("log(Fitness Value)")
    # 处理可能为0或负数的适应度值，避免做对数计算出现除0错误
    valid_fitness_values = np.maximum(fitness_values, 1e-10)
    plt.plot(np.log(valid_fitness_values), "r-", linewidth=2)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
