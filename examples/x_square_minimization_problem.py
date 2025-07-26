import numpy as np

from pypso import plot
from pypso.core import PyPSO, AlgorithmArguments, ProblemType


def x_square_minimization_problem(positions: np.ndarray) -> np.ndarray:
    """
    这个函数定义了一个简单的优化问题：最小化 f(x) = x^2
    该函数接受一个位置数组，并返回每个位置的平方值作为适应度值
    """
    return positions ** 2


if __name__ == "__main__":
    pso_optimizer = PyPSO(AlgorithmArguments(
        num_particles=100,
        num_dimensions=1,
        max_iterations=100,
        position_bound_min=-10,
        position_bound_max=10,
        velocity_bound_max=1,
        inertia_weight_max=2,
        inertia_weight_min=0.5,
        cognitive_coefficient=0.5,
        social_coefficient=0.5,
        fitness_function=lambda positions: x_square_minimization_problem(positions)
    ))
    best_solutions, best_fitness_values = pso_optimizer.start_iterating(ProblemType.MINIMIZATION)
    plot.plot_fitness_curve(best_fitness_values)
