import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from pypso.core import AlgorithmArguments, PyPSO, ProblemType
from pypso.util import terrain, plot

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
        # 初始化每个粒子的最优路径成本
        self.particles_cost: np.ndarray = np.zeros(pso_args.num_particles)

    def plot_map_with_best_path(self) -> None:
        """
        绘制地形和最优路径
        """
        # 在最优路径点中追加终点
        self.best_path_points.append(DESTINATION)
        logger.success(f"PSO算法迭代完成，共找到 {len(self.best_path_points) - 2} 个最优路径点（不包含起点和终点）")

        # 绘制地形
        figure = plt.figure(figsize=(10, 8))
        ax = figure.add_subplot(111, projection="3d")
        surface = ax.plot_surface(self.x_grid, self.y_grid, self.z_grid, cmap="viridis", alpha=0.6)
        figure.colorbar(surface, shrink=0.5, aspect=5)

        # 标记起点和终点
        start_x, start_y, start_z = START_POINT[0], START_POINT[1], START_POINT[2]
        destination_x, destination_y, destination_z = DESTINATION[0], DESTINATION[1], DESTINATION[2]
        ax.scatter(start_x, start_y, start_z, c='green', s=100, marker='o', label='Start Point')
        ax.scatter(destination_x, destination_y, destination_z, c='red', s=100, marker='*', label='Destination')

        # 绘制路径
        np_best_path_points = np.array(self.best_path_points)
        path_x, path_y, path_z = np_best_path_points[:, 0], np_best_path_points[:, 1], np_best_path_points[:, 2]
        ax.plot(path_x, path_y, path_z, 'b-', linewidth=5, label='Best Path')

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
        # PSO算法每迭代一次，就会产生一个当前最优路径点，在初始化时，它是起点，往后每次取最后一个最优路径点
        current_best_point = self.best_path_points[-1]

        # 计算每个粒子到当前点的距离，作为先天性成本
        distances = np.linalg.norm(positions - current_best_point, axis=1)
        self.particles_cost += distances

        # 惩罚机制
        for i, p in enumerate(positions):
            point = np.array(p)
            penalty = 0
            # 碰撞惩罚
            if terrain.is_collision_detected(point, self.x_grid, self.y_grid, self.z_grid):
                logger.warning(f"粒子 {i} 在点 {point} 处与地形发生碰撞")
                penalty += 1000
            # 高度惩罚
            allowed_max_height = (DESTINATION[2] + 1) / 2
            if point[2] > allowed_max_height:
                # 偏离越远惩罚越重
                logger.warning(f"粒子 {i} 在点 {point} 处高度 {point[2]} 超过允许最大高度 {allowed_max_height}")
                penalty += abs(point[2] - allowed_max_height) * 100
            # 惩罚项计入路径成本
            self.particles_cost[i] += penalty

        # 选择总成本最小的粒子作为下一个最优路径点
        best_point_index = np.argmin(self.particles_cost)
        best_point = positions[best_point_index]
        best_x, best_y, best_z = best_point[0], best_point[1], best_point[2]
        self.best_path_points.append((best_x.item(), best_y.item(), best_z.item()))

        return self.particles_cost


if __name__ == "__main__":
    # 控制日志级别
    PyPSO.set_logger_level("INFO")

    # 初始化PSO算法参数
    pso_arguments = AlgorithmArguments(
        num_particles=100,
        num_dimensions=3,
        max_iterations=100,
        position_bounds_min=(0, 0, 0),
        position_bounds_max=(100, 100, DESTINATION[2] + 1),
        velocity_bound_max=1,
        inertia_weight_max=2,
        inertia_weight_min=0.5,
        cognitive_coefficient=0.5,
        social_coefficient=0.5
    )

    # 初始化待优化的路径规划问题
    path_planning_3d = PathPlanning3D(pso_arguments)

    # 初始化PSO优化器
    pso_optimizer = PyPSO(
        args=pso_arguments,
        objective_function=path_planning_3d.compute_path_cost
    )

    # 执行算法迭代
    best_fitness_values = pso_optimizer.start_iterating(
        problem_type=ProblemType.MINIMIZATION,
        dynamic_check_convergence=False
    )

    # 绘制适应度曲线
    plot.plot_fitness_curve(fitness_values=best_fitness_values, sup_title="3D Path Planning")

    # 绘制最优路径
    path_planning_3d.plot_map_with_best_path()
