from enum import Enum, unique
from typing import Callable

import numpy as np
from loguru import logger
from pydantic import BaseModel


class AlgorithmArguments(BaseModel):
    """定义粒子群优化算法（Particle Swarm Optimization）核心参数"""

    num_particles: int
    """粒子数量"""

    num_dimensions: int
    """粒子搜索的空间维度，也就是待优化问题的自变量个数"""

    max_iterations: int
    """最大迭代次数"""

    inertia_weight_max: float
    """最大惯性权重，在迭代过程中会使用线性递减来动态调整，但不会大于该值"""

    inertia_weight_min: float
    """最小惯性权重，在迭代过程中会使用线性递增来动态调整，但不会小于该值"""

    cognitive_coefficient: float
    """认知系数C1"""

    social_coefficient: float
    """社会系数C2"""

    position_bound_min: float
    """粒子位置下界，限制其搜索空间范围，也就是限制自变量的取值范围，对于无约束的问题，可以设置一个很小的值"""

    position_bound_max: float
    """粒子位置上界，限制其搜索空间范围，也就是限制自变量的取值范围，对于无约束的问题，可以设置一个很大的值"""

    velocity_bound_max: float
    """速度上界，因为速度是矢量，所以上界取反方向就可以得到速度下界，目的是为了平衡算法的探索能力和开发能力"""

    fitness_function: Callable[[np.ndarray], np.ndarray]
    """适应度函数，输入是个体位置数组（可能是多维），输出是适应度数组（只会是一维）"""

    auto_plot_fitness_curve: bool = True
    """是否自动绘制适应度曲线"""


@unique
class ProblemType(Enum):
    """定义问题类型枚举"""

    MINIMIZATION_PROBLEM = 1
    """最小化问题"""

    MAXIMIZATION_PROBLEM = 2
    """最大化问题"""


class PyPSO:
    """粒子群优化算法（Particle Swarm Optimization）核心实现"""

    def __init__(self, args: AlgorithmArguments):
        """算法初始化"""

        logger.debug(f"初始化PSO算法，使用以下参数：{args.model_dump_json(indent=4, exclude={'fitness_function'})}")

        self.num_particles = args.num_particles
        self.num_dimensions = args.num_dimensions
        self.max_iterations = args.max_iterations
        self.inertia_weight_max = args.inertia_weight_max
        self.inertia_weight_min = args.inertia_weight_min
        self.cognitive_coefficient = args.cognitive_coefficient
        self.social_coefficient = args.social_coefficient
        self.position_bound_min = args.position_bound_min
        self.position_bound_max = args.position_bound_max
        self.velocity_bound_max = args.velocity_bound_max
        self.fitness_function = args.fitness_function
        self.auto_plot_fitness_curve = args.auto_plot_fitness_curve

        self.iteration_history = []  # 迭代历史，用于记录每次迭代的全局最佳适应度，方便绘制迭代曲线

        shape = (args.num_particles, args.num_dimensions)
        logger.debug(f"定义矩阵大小为：{args.num_particles} x {args.num_dimensions}")

        self.particles_position = np.random.uniform(args.position_bound_min, args.position_bound_max, shape)
        logger.debug(f"随机初始化粒子位置：{self.particles_position}")

        self.particles_velocity = np.random.uniform(-args.velocity_bound_max, args.velocity_bound_max, shape)
        logger.debug(f"随机初始化粒子速度：{self.particles_velocity}")

        self.particles_best_position = self.particles_position.copy()
        logger.debug(f"初始化个体最优位置（当前位置）：{self.particles_best_position}")

        self.particles_best_fitness = args.fitness_function(self.particles_position)
        logger.debug(f"初始化个体最优适应度（需要结合适应度函数做计算）：{self.particles_best_fitness}")

        best_particle_index = np.argmin(self.particles_best_fitness)
        logger.debug(f"计算个体最优适应度最小值的下标：{best_particle_index}")

        self.global_best_position = self.particles_best_position[best_particle_index]
        logger.debug(f"初始化全局最优位置（以最小下标取个体最优位置的值）：{self.global_best_position}")

        self.global_best_fitness = self.particles_best_fitness[best_particle_index]
        logger.debug(f"初始化全局最优适应度（以最小下标取个体最优适应度的值）：{self.global_best_fitness}")

        logger.success("PSO算法初始化成功")

    def _update_all_particles_velocity(self) -> None:
        """向量化更新所有个体的速度"""

        cognitive_coefficient_random_weight = np.random.uniform(0, 1, (self.num_particles, 1))
        social_coefficient_random_weight = np.random.uniform(0, 1, (self.num_particles, 1))

        cognitive = cognitive_coefficient_random_weight * (self.particles_best_position - self.particles_position)
        social = social_coefficient_random_weight * (self.global_best_position - self.particles_position)

        # 速度更新公式
        self.particles_velocity = (
                self.inertia_weight * self.particles_velocity +
                self.cognitive_coefficient * cognitive +
                self.social_coefficient * social
        )

        # 限制速度边界
        self.particles_velocity = np.clip(self.particles_velocity, -self.velocity_bound_max, self.velocity_bound_max)

    def _update_all_particle_position(self) -> None:
        """向量化更新所有个体的位置"""
        next_particles_position = self.particles_position + self.particles_velocity
        self.particles_position = np.clip(next_particles_position, self.position_bound_min, self.position_bound_max)

    def _update_best_fitness_for_minimization_problem(self) -> None:
        """向量化更新个体和全局最优适应度，该方法适用于求解最小化问题"""

        # 计算所有粒子的当前适应度
        computed_particles_fitness = self.fitness_function(self.particles_position)
        # 找出哪些粒子的当前适应度优于个体最优
        improved_mask = computed_particles_fitness < self.particles_best_fitness
        # 更新个体最优位置和适应度
        self.particles_best_position[improved_mask] = self.particles_position[improved_mask]
        self.particles_best_fitness[improved_mask] = computed_particles_fitness[improved_mask]
        # 更新全局最优位置和适应度
        best_particle_index = np.argmin(self.particles_best_fitness)
        candidate_global_fitness = self.particles_best_fitness[best_particle_index]
        if candidate_global_fitness < self.global_best_fitness:
            self.global_best_position = self.particles_best_position[best_particle_index]
            self.global_best_fitness = candidate_global_fitness

    def _update_best_fitness_for_maximization_problem(self) -> None:
        """向量化更新个体和全局最优适应度，该方法适用于求解最大化问题"""

        # 计算所有粒子的当前适应度
        computed_particles_fitness = self.fitness_function(self.particles_position)
        # 找出哪些粒子的当前适应度优于个体最优
        improved_mask = computed_particles_fitness > self.particles_best_fitness
        # 更新个体最优位置和适应度
        self.particles_best_position[improved_mask] = self.particles_position[improved_mask]
        self.particles_best_fitness[improved_mask] = computed_particles_fitness[improved_mask]
        # 更新全局最优位置和适应度
        best_particle_index = np.argmin(self.particles_best_fitness)
        candidate_global_fitness = self.particles_best_fitness[best_particle_index]
        if candidate_global_fitness > self.global_best_fitness:
            self.global_best_position = self.particles_best_position[best_particle_index]
            self.global_best_fitness = candidate_global_fitness

    def _compute_linear_decrease_inertia_weight(self, iteration: int, max_iterations: int) -> None:
        """
        动态计算惯性权重，使用线性递减规则

        Args:
            iteration (int): 当前迭代次数
            max_iterations (int): 最大迭代次数

        Returns:
            None
        """
        inertia_weight_range_value = self.inertia_weight_max - self.inertia_weight_min
        self.inertia_weight = self.inertia_weight_max - inertia_weight_range_value * (iteration / max_iterations)

    def _check_fitness_converged(self, problem_type: ProblemType) -> bool:
        """检查全局最佳适应度值是否已经收敛"""

        # 最小化问题
        if problem_type is ProblemType.MINIMIZATION_PROBLEM:
            return abs(self.global_best_fitness) < 1e-6

        # 最大化问题
        if problem_type is ProblemType.MAXIMIZATION_PROBLEM:
            # 判断适应度变化率是否已经足够小
            length = len(self.iteration_history)
            if length > 1:
                global_best_fitness_bias = self.iteration_history[length - 1] - self.iteration_history[length - 2]
                return abs(global_best_fitness_bias) < 1e-6

        return False

    def _hook_on_all_iterations_finished(self) -> None:
        """全部迭代结束以后的hook函数，可以用于做一些后续工作，但又不用和迭代逻辑耦合"""
        if self.auto_plot_fitness_curve:
            from pypso import plot
            plot.plot_fitness_curve(self.iteration_history)

    def start_iterating(self, problem_type: ProblemType = ProblemType.MINIMIZATION_PROBLEM) -> None:
        """
        开始执行算法迭代

        Args:
            problem_type (ProblemType): 待优化的问题类型，可以是最小化问题，也可以是最大化问题

        Returns:
            None
        """

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"第{iteration}次迭代开始")

            # 适应度收敛判断
            if self._check_fitness_converged(problem_type):
                logger.info(f"适应度已收敛，可以提前结束迭代")
                break

            # 动态更新惯性权重
            self._compute_linear_decrease_inertia_weight(iteration, self.max_iterations)

            # 更新个体速度和位置，使用向量化更新，可以比循环语句有更好的性能
            self._update_all_particles_velocity()
            self._update_all_particle_position()

            # 更新最佳适应度，同样使用向量化更新
            if problem_type is ProblemType.MAXIMIZATION_PROBLEM:
                self._update_best_fitness_for_maximization_problem()
            else:
                self._update_best_fitness_for_minimization_problem()

            # 记录每次迭代的全局最佳适应度，方便绘制迭代曲线
            self.iteration_history.append(self.global_best_fitness.item())

            logger.info(f"第{iteration}次迭代结束，全局最佳适应度：{self.global_best_fitness.item():.6f}")

        # 全部迭代结束后执行一些hook函数
        self._hook_on_all_iterations_finished()
