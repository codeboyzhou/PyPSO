import numpy as np
from scipy.ndimage import gaussian_filter


def generate_simulated_mountain_peaks(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    peaks: list[tuple[float, float, float, float]]
) -> np.ndarray:
    """
    生成模拟山峰的地形图在Z网格的坐标

    Args:
        x_grid (np.ndarray): X平面网格
        y_grid (np.ndarray): Y平面网格
        peaks (list[tuple[float, float, float, float]]): 山峰参数列表
            每个元组包含 (center_x, center_y, amplitude, width)
            center_x: 山峰中心的X坐标
            center_y: 山峰中心的Y坐标
            amplitude: 山峰的高度
            width: 山峰的宽度

    Returns:
        np.ndarray: 生成的地形图在Z网格的坐标
    """
    # 生成基础地形
    z_grid = np.zeros_like(x_grid)

    # 添加山峰
    for (center_x, center_y, amplitude, width) in peaks:
        z_grid += amplitude * np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * width ** 2))

    # 添加一些随机噪声和基础波动，增强山峰的真实性
    z_grid += 0.2 * np.sin(0.5 * np.sqrt(x_grid ** 2 + y_grid ** 2)) + 0.1 * np.random.normal(size=x_grid.shape)

    # 使用高斯滤波，保持山峰独立性的同时也保证平滑性
    z_grid = gaussian_filter(z_grid, sigma=3)

    return z_grid
