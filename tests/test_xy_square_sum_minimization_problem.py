import numpy as np

from pypso.core import PyPSO, AlgorithmArguments, ProblemType
from pypso.util import plot


def xy_square_sum_minimization_problem(positions: np.ndarray) -> np.ndarray:
    """
    这个函数定义了一个简单的优化问题：最小化 f(x, y) = x^2 + y^2
    该函数接受一个位置数组，并返回每个位置的平方和作为适应度值
    """
    return np.sum(positions ** 2, axis=1)


def test_xy_square_sum_minimization_problem():
    """
    测试 xy_square_sum_minimization_problem 函数
    该函数用于测试 PSO 算法在二维最小化问题上的表现
    """
    # 初始化PSO算法参数
    pso_arguments = AlgorithmArguments(
        num_particles=100,
        num_dimensions=2,
        max_iterations=100,
        position_bounds_min=-10,
        position_bounds_max=10,
        velocity_bound_max=1,
        inertia_weight_max=2,
        inertia_weight_min=0.5,
        cognitive_coefficient=0.5,
        social_coefficient=0.5
    )

    # 初始化PSO优化器
    pso_optimizer = PyPSO(args=pso_arguments, objective_function=xy_square_sum_minimization_problem)

    # 执行算法迭代
    best_fitness_values = pso_optimizer.start_iterating(ProblemType.MINIMIZATION)

    # 绘制适应度曲线
    plot.plot_fitness_curve(
        fitness_values=best_fitness_values,
        sup_title=r"f(x) = $x^{2} + y^{2}$",
        block=False
    )
