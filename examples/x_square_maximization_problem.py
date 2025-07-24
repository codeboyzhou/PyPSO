from pypso.core import PyPSO, AlgorithmArguments, ProblemType

if __name__ == "__main__":
    PyPSO(AlgorithmArguments(
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
        fitness_function=lambda x: x ** 2,
        auto_plot_fitness_curve=True,
    )).start_iterating(problem_type=ProblemType.MAXIMIZATION_PROBLEM)
