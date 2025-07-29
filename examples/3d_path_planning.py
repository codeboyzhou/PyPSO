import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.interpolate import make_interp_spline

from pypso.core import AlgorithmArguments, PyPSO, ProblemType
from pypso.util import terrain, plot, ndarrays

# 定义模拟山峰的参数
PEAKS = [
    # center_x, center_y, amplitude, width
    (20, 20, 6, 6),
    (20, 60, 7, 7),
    (60, 20, 5, 8),
    (80, 60, 5, 8),
]

# 定义路径起点和终点坐标
START_POINT = (0.0, 0.0, 0.0)
DESTINATION = (80.0, 80.0, 1.0)


class PathPlanning3D:

    def __init__(self, pso_args: AlgorithmArguments):
        # 生成坐标网格
        x = np.linspace(pso_args.position_bounds_min[0], pso_args.position_bounds_max[0])
        y = np.linspace(pso_args.position_bounds_min[1], pso_args.position_bounds_max[1])
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        self.z_grid = terrain.generate_simulated_mountain_peaks(self.x_grid, self.y_grid, PEAKS)

        # 初始化最优路径点
        self.best_path_points: list[tuple[float, float, float]] = [START_POINT]

    def plot_map_with_best_path(self, mark_waypoints: bool = True) -> None:
        """
        绘制地形和最优路径
        """
        # 在最优路径点中追加终点
        self.best_path_points.append(DESTINATION)
        # 对路径点按坐标进行排序
        self.best_path_points.sort(key=lambda sorting_point: sorting_point[0] + sorting_point[1] + sorting_point[2])
        logger.success(f"共找到 {len(self.best_path_points)} 个最优路径点：{self.best_path_points}")

        # 绘制地形
        figure = plt.figure(figsize=(10, 8))
        ax = figure.add_subplot(111, projection="3d")
        surface = ax.plot_surface(self.x_grid, self.y_grid, self.z_grid, cmap="viridis", alpha=0.6)
        figure.colorbar(surface, shrink=0.5, aspect=5)

        # 标记路径点
        for i, point in enumerate(self.best_path_points):
            px, py, pz = point[0], point[1], point[2]
            # 起点
            if i == 0:
                ax.scatter(px, py, pz, c="green", s=100, marker="o", label="Start Point")
            # 终点
            elif i == len(self.best_path_points) - 1:
                ax.scatter(px, py, pz, c="red", s=100, marker="*", label="Destination")
            # 途经点
            elif mark_waypoints:
                ax.scatter(px, py, pz, c="orange", s=100, marker="^", label="Waypoint" if i == 1 else None)
                ax.text(px + 0.2, py + 0.2, z=pz + 0.2, color="red", s=f"P{i}", fontsize=10)

        # 使用三阶B样条曲线绘制平滑路径
        np_best_path_points = np.array(self.best_path_points)
        x, y, z = np_best_path_points[:, 0], np_best_path_points[:, 1], np_best_path_points[:, 2]
        # 计算路径点的参数化变量
        t_for_spline = np.arange(len(x))
        # 创建三阶B样条曲线（k=3表示三阶）
        spline = make_interp_spline(t_for_spline, np.column_stack((x, y, z)), k=3)
        # 生成平滑路径点，使用参数化变量进行插值，可以根据需要调整点的数量
        smooth_path = spline(np.linspace(t_for_spline.min(), t_for_spline.max(), 100))
        x, y, z = smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2]
        ax.plot(x, y, z, "b-", linewidth=5, label="Best Path")

        # 设置坐标轴信息，elev参数控制仰角，azim参数控制方位角
        ax.view_init(elev=30, azim=240)
        ax.set_title("3D Path Planning")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Height")
        ax.legend()

        plt.tight_layout()
        plt.show()

    def compute_path_cost(self, positions: np.ndarray) -> np.ndarray:
        """
        适应度函数：计算路径成本

        Args:
            positions (np.ndarray): 所有粒子的位置向量，形状为 (num_particles, num_dimensions)

        Returns:
            路径成本大小，数据格式为 np.ndarray，形状为 (n, 1)
        """
        # 初始化粒子路径成本
        particles_cost = np.zeros(positions.shape[0])

        # 定义惩罚权重
        iteration = len(self.best_path_points)
        height_diff_weight = 0.5 + 0.1 * iteration
        smoothness_weight = 1.0 + 0.1 * iteration
        distances_to_line_weight = 10 * np.exp(0.01 * iteration)
        distances_to_current_weight = 0.5 * np.exp(-0.01 * iteration)
        distances_to_destination_weight = 0.05 * iteration

        # 高度偏离惩罚，已做归一化处理
        height_diffs = np.abs(positions[:, 2] - DESTINATION[2])
        particles_cost += ndarrays.normalize(height_diffs) * height_diff_weight

        # 路径平滑性惩罚，已做归一化处理
        if len(self.best_path_points) > 1:
            # 计算相邻路径点的方向向量余弦值
            current_point = np.array(self.best_path_points[-1])
            previous_point = np.array(self.best_path_points[-2])
            best_path_directions = positions - current_point
            previous_direction = current_point - previous_point
            cos_angles = ndarrays.cos_angles(best_path_directions, previous_direction)
            # 余弦值越接近1，说明余弦夹角越小，路径越平滑，因此惩罚值为1减去余弦值
            smoothness = 1 - cos_angles
            particles_cost += ndarrays.normalize(smoothness) * smoothness_weight

        # 偏离起点到终点所在直线惩罚，已做归一化处理
        distances_to_line = ndarrays.point_to_line_distance(positions, np.array(START_POINT), np.array(DESTINATION))
        particles_cost += ndarrays.normalize(distances_to_line) * distances_to_line_weight

        # 距离当前点过远惩罚，已做归一化处理
        distances_to_current = np.linalg.norm(positions - self.best_path_points[-1], axis=1)
        particles_cost += ndarrays.normalize(distances_to_current) * distances_to_current_weight

        # 距离终点过远惩罚，已做归一化处理
        distances_to_destination = np.linalg.norm(positions - DESTINATION, axis=1)
        particles_cost += ndarrays.normalize(distances_to_destination) * distances_to_destination_weight

        # 选择成本最小的点
        best_point_index = np.argmin(particles_cost)
        best_point = positions[best_point_index]

        # 碰撞点纠偏
        while terrain.is_collision_detected(np.array(best_point), self.x_grid, self.y_grid, self.z_grid):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 -x 方向纠偏")
            best_point[0] -= 1
            if best_point[0] < 0:
                best_point[0] = 0
        while terrain.is_collision_detected(np.array(best_point), self.x_grid, self.y_grid, self.z_grid):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 +x 方向纠偏")
            best_point[0] += 1
            if best_point[0] > 100:
                best_point[0] = 100
        while terrain.is_collision_detected(np.array(best_point), self.x_grid, self.y_grid, self.z_grid):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 -y 方向纠偏")
            best_point[1] -= 1
            if best_point[1] < 0:
                best_point[1] = 0
        while terrain.is_collision_detected(np.array(best_point), self.x_grid, self.y_grid, self.z_grid):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 +y 方向纠偏")
            best_point[1] += 1
            if best_point[1] > 100:
                best_point[1] = 100

        best_x, best_y, best_z = best_point[0], best_point[1], best_point[2]
        self.best_path_points.append((best_x.item(), best_y.item(), best_z.item()))

        return particles_cost


if __name__ == "__main__":
    # 控制日志级别
    PyPSO.set_logger_level("INFO")

    # 初始化PSO算法参数
    pso_arguments = AlgorithmArguments(
        num_particles=100,
        num_dimensions=3,
        max_iterations=100,
        position_bounds_min=(0, 0, 0),
        position_bounds_max=(100, 100, DESTINATION[2]),
        velocity_bound_max=1,
        inertia_weight_max=2,
        inertia_weight_min=0.5,
        cognitive_coefficient=1.6,
        social_coefficient=1.8
    )

    # 初始化待优化的路径规划问题
    path_planning_3d = PathPlanning3D(pso_arguments)

    # 初始化PSO优化器
    pso_optimizer = PyPSO(
        args=pso_arguments,
        objective_function=path_planning_3d.compute_path_cost
    )

    # 执行算法迭代
    best_fitness_values = pso_optimizer.start_iterating(ProblemType.MINIMIZATION)

    # 绘制适应度曲线
    plot.plot_fitness_curve(fitness_values=best_fitness_values, block=False)

    # 绘制最优路径
    path_planning_3d.plot_map_with_best_path()
