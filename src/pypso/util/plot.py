import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_curve(fitness_values: list[float], sup_title: str = None, block: bool = True) -> None:
    """
    绘制适应度函数曲线和对应的对数函数曲线

    Args:
        fitness_values (list[float]): 适应度值，浮点数，一维数组
        sup_title (str): 图形的总标题，默认为 None
        block (bool): 是否阻塞显示图形窗口，默认为 True，如果为 False，则不会阻塞

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    plt.suptitle(sup_title)
    plt.title("Fitness Function Curve")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Fitness Value")
    plt.plot(fitness_values, "b-", linewidth=2)
    plt.grid(True)
    plt.tight_layout()

    if block:
        plt.show()
    else:
        plt.draw()
        plt.pause(1)
