import numpy as np
from loguru import logger
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
            amplitude: 山峰的高度系数
            width: 山峰的宽度

    Returns:
        np.ndarray: 生成的地形图在Z网格的坐标
    """
    logger.success("开始生成模拟山峰地形")

    # 生成基础地形
    z_grid = np.zeros_like(x_grid)

    # 添加山峰
    for (center_x, center_y, amplitude, width) in peaks:
        z_grid += amplitude * np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * width ** 2))

    # 添加一些随机噪声和基础波动，增强山峰的真实性
    z_grid += 0.2 * np.sin(0.5 * np.sqrt(x_grid ** 2 + y_grid ** 2)) + 0.1 * np.random.normal(size=x_grid.shape)

    # 使用高斯滤波，保持山峰独立性的同时也保证平滑性
    z_grid = gaussian_filter(z_grid, sigma=3)

    logger.success(f"山峰 {peaks} 已添加到地形图中，最大高度：{np.max(z_grid)}")

    return z_grid


def is_collision_detected(
    point: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray
) -> bool:
    """
    检测空间中的某个点 (x, y, z) 是否与高度场发生了碰撞

    Args:
        point (np.ndarray): 待检测的点的坐标 (x, y, z)
        x_grid (np.ndarray): X平面网格
        y_grid (np.ndarray): Y平面网格
        z_grid (np.ndarray): Z平面网格

    Returns:
        bool: 如果发生碰撞则返回True，否则返回False
    """
    x, y, z = point[0], point[1], point[2]

    # 边界值处理，高度为0，无需考虑碰撞
    if z == 0:
        return False

    # 获取网格的实际坐标值范围
    x_vals = np.unique(x_grid)
    y_vals = np.unique(y_grid)
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # 超出地形范围视为碰撞
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return True

    # 计算实际坐标步长并获取网格索引
    x_step = x_vals[1] - x_vals[0]
    y_step = y_vals[1] - y_vals[0]
    x_index = np.clip(int((x - x_min) / x_step), 0, len(x_vals) - 1)
    y_index = np.clip(int((y - y_min) / y_step), 0, len(y_vals) - 1)

    # 获取地形高度并增加安全余量（10%地形高度）
    terrain_z = z_grid[y_index, x_index]  # 注意meshgrid的索引顺序是(y, x)
    safety_margin = 0.1 * terrain_z  # 增加10%的安全高度避免贴地穿模

    # 点高度低于地形高度+安全余量则判定为碰撞
    result = z < (terrain_z + safety_margin)

    return result.item()
