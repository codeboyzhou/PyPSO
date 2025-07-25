import matplotlib.pyplot as plt
import numpy as np
import scipy

from pypso.core import PyPSO, AlgorithmArguments, ProblemType


def gaussian_peak(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    center_x: float,
    center_y: float,
    amplitude: float,
    width: float
) -> np.ndarray:
    """
    生成山峰的形状

    Args:
        x_grid (np.ndarray): X平面网格
        y_grid (np.ndarray): Y平面网格
        center_x (float): 山峰中心点的X坐标
        center_y (float): 山峰中心点的Y坐标
        amplitude (float): 山峰振幅
        width (float): 山峰宽度

    Returns:
        山峰在Z网格的坐标
    """
    return amplitude * np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * width ** 2))


def plot_map_with_line(start_point: tuple, destination: tuple, path_points: np.ndarray = None) -> None:
    """
    绘制基础地形和山峰，并根据给定的起点、终点、路径点数组绘制一条路径

    Args:
        start_point (tuple(float, float, float)): 起点坐标(x, y, z)
        destination (tuple(float, float, float)): 终点坐标(x, y, z)
        path_points (np.ndarray): 路径点坐标数组，形状为(n, 3)
    """
    # 生成网格
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    x_grid, y_grid = np.meshgrid(x, y)

    # 定义山峰的位置和参数
    peaks = [
        # center_x, center_y, amplitude, width
        (20, 20, 6, 6),
        (20, 60, 7, 7),
        (60, 20, 5, 8),
        (80, 60, 5, 8),
    ]

    # 生成基础地形
    z_grid = np.zeros_like(x_grid)

    # 添加山峰
    for (center_x, center_y, amplitude, width) in peaks:
        z_grid += gaussian_peak(x_grid, y_grid, center_x, center_y, amplitude, width)

    # 添加一些随机噪声和基础波动，增强山峰的真实性
    z_grid += 0.2 * np.sin(0.5 * np.sqrt(x_grid ** 2 + y_grid ** 2)) + 0.1 * np.random.normal(size=x_grid.shape)

    # 使用高斯滤波，保持山峰独立性的同时也保证平滑性
    z_grid = scipy.ndimage.gaussian_filter(z_grid, sigma=3)

    # 开始绘制地形和山峰
    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(111, projection="3d")
    surface = ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", alpha=0.6)
    figure.colorbar(surface, shrink=0.5, aspect=5)

    # 标记起点和终点
    start_point_x, start_point_y, start_point_z = start_point[0], start_point[1], start_point[2]
    destination_x, destination_y, destination_z = destination[0], destination[1], destination[2]
    ax.scatter(start_point_x, start_point_y, start_point_z, c='green', s=100, marker='o', label='Start Point')
    ax.scatter(destination_x, destination_y, destination_z, c='red', s=100, marker='*', label='Destination')

    # 如果提供了路径点，则绘制完整路径
    if path_points is not None and len(path_points) > 0:
        # 将起点、路径点、终点连接起来
        path_points_x, path_points_y, path_points_z = path_points[:, 0], path_points[:, 1], path_points[:, 2]
        all_points_x = [start_point_x] + list(path_points_x) + [destination_x]
        all_points_y = [start_point_y] + list(path_points_y) + [destination_y]
        all_points_z = [start_point_z] + list(path_points_z) + [destination_z]

        # 绘制完整路径
        ax.plot(all_points_x, all_points_y, all_points_z, 'b-', linewidth=5, label='Target Path')

        # 标记路径点
        ax.scatter(path_points_x, path_points_y, path_points_z, c='orange', s=50, marker='o', label='Path Points')
    else:
        # 只绘制起点到终点的直线
        line_x = [start_point_x, destination_x]
        line_y = [start_point_y, destination_y]
        line_z = [start_point_z, destination_z]
        ax.plot(line_x, line_y, line_z, 'g-', linewidth=5, label='Target Path')

    # 设置坐标轴信息
    ax.set_title("3D Path Planning")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Height")
    ax.legend()
    # elev参数控制仰角，azim参数控制方位角
    ax.view_init(elev=30, azim=240)

    plt.tight_layout()
    plt.show()


def three_dimensional_path_planning_problem(positions: np.ndarray) -> np.ndarray:
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    return np.sqrt((x - 50) ** 2 + (y - 50) ** 2 + (z - 50) ** 2)


if __name__ == "__main__":
    PyPSO(AlgorithmArguments(
        num_particles=100,
        num_dimensions=3,
        max_iterations=100,
        position_bound_min=-10,
        position_bound_max=10,
        velocity_bound_max=1,
        inertia_weight_max=2,
        inertia_weight_min=0.5,
        cognitive_coefficient=0.5,
        social_coefficient=0.5,
        fitness_function=lambda positions: three_dimensional_path_planning_problem(positions)
    )).start_iterating(
        problem_type=ProblemType.MINIMIZATION_PROBLEM,
        auto_plot_fitness_curve=True
    )

    plot_map_with_line(
        start_point=(0, 0, 0),
        destination=(80, 80, 2),
        path_points=np.array([
            [10, 0, 2],
            [20, 0, 2],
            [30, 0, 2],
            [40, 0, 2],
            [40, 40, 2],
            [50, 50, 2],
            [60, 60, 2],
            [70, 70, 2],
        ])
    )
