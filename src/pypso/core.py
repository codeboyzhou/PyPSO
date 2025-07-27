import random
from enum import Enum, unique
from typing import Callable

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pypso.util import checker


class AlgorithmArguments(BaseModel):
    """定义粒子群优化算法（Particle Swarm Optimization）核心参数"""

    num_particles: int
    """粒子群规模，即粒子个数"""

    num_dimensions: int
    """待优化问题的维度，即问题涉及到的自变量个数"""

    max_iterations: int
    """最大迭代次数"""

    inertia_weight_max: float
    """最大惯性权重，用于实现惯性权重的线性递减，初始值较大可以增加算法的探索能力"""

    inertia_weight_min: float
    """最小惯性权重，用于实现惯性权重的线性递减，最终值较小可以增加算法的开发能力"""

    cognitive_coefficient: float
    """认知系数C1"""

    social_coefficient: float
    """社会系数C2"""

    position_bounds_min: tuple[float, ...]
    """
    粒子位置下界，即允许自变量可取的最小值
    tuple类型，有几个自变量，就有几个元素
    对于无约束的问题，可以设置一个很小的值
    """

    position_bounds_max: tuple[float, ...]
    """
    粒子位置上界，即允许自变量可取的最大值
    tuple类型，有几个自变量，就有几个元素
    对于无约束的问题，可以设置一个很大的值
    """

    velocity_bound_max: float
    """速度上界，因为速度是矢量，所以上界取反方向就可以得到速度下界，目的是为了平衡算法的探索能力和开发能力"""


@unique
class ProblemType(Enum):
    """定义问题类型枚举"""

    MINIMIZATION = 1
    """最小化问题"""

    MAXIMIZATION = 2
    """最大化问题"""


class PyPSO:
    """粒子群优化算法（Particle Swarm Optimization）核心实现"""

    def __init__(self, args: AlgorithmArguments, objective_function: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        算法初始化

        Args:
            args (AlgorithmArguments): 算法核心参数
            objective_function (Callable[[np.ndarray], np.ndarray]): 待优化问题的目标函数
                接受一个形状为 (num_particles, num_dimensions) 的 numpy 数组
                返回一个形状为 (num_particles, 1) 的 numpy 数组，表示适应度值

        Returns:
            None
        """

        logger.debug(f"初始化PSO算法，使用以下参数：{args.model_dump_json(indent=4)}")

        # 算法核心参数
        self.num_particles = args.num_particles
        self.num_dimensions = args.num_dimensions
        self.max_iterations = args.max_iterations
        self.inertia_weight_max = args.inertia_weight_max
        self.inertia_weight_min = args.inertia_weight_min
        self.cognitive_coefficient = args.cognitive_coefficient
        self.social_coefficient = args.social_coefficient
        self.position_bounds_min = args.position_bounds_min
        self.position_bounds_max = args.position_bounds_max
        self.velocity_bound_max = args.velocity_bound_max

        # 待优化问题的目标函数
        self.objective_function = objective_function

        shape = (args.num_particles, args.num_dimensions)
        logger.debug(f"定义矩阵大小为：{args.num_particles} x {args.num_dimensions}")

        self.particles_position = np.random.uniform(args.position_bounds_min, args.position_bounds_max, shape)
        logger.debug(f"随机初始化粒子位置：{self.particles_position}")

        self.particles_velocity = np.random.uniform(-args.velocity_bound_max, args.velocity_bound_max, shape)
        logger.debug(f"随机初始化粒子速度：{self.particles_velocity}")

        self.particles_best_position = self.particles_position.copy()
        logger.debug(f"初始化个体最优位置（当前位置）：{self.particles_best_position}")

        self.particles_best_fitness = objective_function(self.particles_position)
        logger.debug(f"初始化个体最优适应度（需要结合目标函数做计算）：{self.particles_best_fitness}")

        bast_particle_index = random.randint(0, self.num_particles - 1)
        self.global_best_position = self.particles_best_position[bast_particle_index]
        self.global_best_fitness = self.particles_best_fitness[bast_particle_index]
        logger.debug(f"初始化全局最优位置（随机假设一个个体的位置）：{self.global_best_position}")
        logger.debug(f"初始化全局最优适应度（随机假设一个个体的适应度）：{self.global_best_fitness}")

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
        self.particles_position = np.clip(next_particles_position, self.position_bounds_min, self.position_bounds_max)

    def _update_best_fitness(self, problem_type: ProblemType) -> None:
        """
        向量化更新个体和全局最优适应度
        
        Args:
            problem_type (ProblemType): 问题类型（最小化或最大化）
        """

        # 计算所有粒子的当前适应度
        computed_particles_fitness = self.objective_function(self.particles_position)

        # 根据问题类型选择比较操作符和最优个体下标索引函数
        if problem_type is ProblemType.MINIMIZATION:
            compare = lambda x, y: x < y
            compute_best_particle_index = np.argmin
        else:
            compare = lambda x, y: x > y
            compute_best_particle_index = np.argmax

        # 找出哪些粒子的当前适应度优于个体最优
        improved_mask = compare(computed_particles_fitness, self.particles_best_fitness)

        # 更新个体最优位置和适应度
        self.particles_best_position[improved_mask] = self.particles_position[improved_mask]
        self.particles_best_fitness[improved_mask] = computed_particles_fitness[improved_mask]

        # 更新全局最优位置和适应度
        best_particle_index = compute_best_particle_index(self.particles_best_fitness)
        candidate_global_fitness = self.particles_best_fitness[best_particle_index]

        if compare(candidate_global_fitness, self.global_best_fitness):
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

    def start_iterating(
        self,
        problem_type: ProblemType,
        dynamic_check_convergence: bool = True
    ) -> list[float]:
        """
        开始执行算法迭代

        Args:
            problem_type (ProblemType): 待优化的问题类型，可以是最小化问题，也可以是最大化问题
            dynamic_check_convergence (bool): 是否动态检查适应度收敛，默认为True

        Returns:
            None
        """
        best_fitness_values = []  # 每次迭代后的最优适应度，全部记录下来用于绘制迭代曲线

        for iteration in range(1, self.max_iterations + 1):
            logger.debug(f"第{iteration}次迭代开始")

            # 检查适应度值是否提前收敛
            if dynamic_check_convergence and checker.is_converged(best_fitness_values):
                logger.debug(f"适应度已收敛，可以提前结束迭代，当前全局最优适应度：{self.global_best_fitness.item():.6f}")
                break

            # 动态更新惯性权重
            self._compute_linear_decrease_inertia_weight(iteration, self.max_iterations)

            # 更新个体速度和位置，使用向量化更新，可以比循环语句有更好的性能
            self._update_all_particles_velocity()
            self._update_all_particle_position()

            # 更新最优适应度，同样使用向量化更新
            self._update_best_fitness(problem_type)

            # 记录每次迭代的全局最优适应度，方便绘制迭代曲线
            best_fitness_values.append(self.global_best_fitness.item())

            logger.debug(f"第{iteration}次迭代结束，全局最优适应度：{self.global_best_fitness.item():.6f}")

        return best_fitness_values
