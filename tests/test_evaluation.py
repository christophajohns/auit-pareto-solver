"""Test functions for AUIT.py"""

# Load the evaluation modules
import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

import experiments.user
import experiments.random_solver
import experiments.pareto_solver
import experiments.weighted_sum_solver
import experiments.problem
import AUIT
import numpy as np

# Hyperparameters
EYE_POSITION = AUIT.networking.element.Position(x=0.0, y=0.0, z=0.0)
SHOULDER_JOINT_POSITION = AUIT.networking.element.Position(x=0.0, y=-0.3, z=0.0)
ARM_LENGTH = 3.0

def get_layout_with_element_at_eye_level():
    """Returns a layout with a UI element at eye level."""
    # Calculate the x-position of the element at eye level
    # in arm's length from the shoulder joint
    y_distance_from_eye_to_shoulder = abs(EYE_POSITION.y - SHOULDER_JOINT_POSITION.y)
    element_x_position = (ARM_LENGTH**2 - y_distance_from_eye_to_shoulder**2) ** 0.5
    # print("Element x position: {}".format(element_x_position))

    # Define test element for cost evaluation at eye level
    element_at_eye_level = AUIT.networking.element.Element(
        id="test_element",
        position=AUIT.networking.element.Position(x=element_x_position, y=EYE_POSITION.y, z=0.0),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    layout = AUIT.networking.layout.Layout(
        items=[element_at_eye_level],
    )
    return layout

def get_layout_with_element_at_waist_level():
    """Returns a layout with a UI element at waist level."""
    # Define test element for cost evaluation at waist level
    element_at_waist_level = AUIT.networking.element.Element(
        id="test_element",
        position=AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x, y=SHOULDER_JOINT_POSITION.y - ARM_LENGTH, z=SHOULDER_JOINT_POSITION.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    layout = AUIT.networking.layout.Layout(
        items=[element_at_waist_level],
    )
    return layout

def test_scenario_1_utility():
    """Test utility function for scenario 1."""
    print("Testing utility for scenario 1...")

    # Get utility function
    get_utility = experiments.user.get_utility_function(
        objectives=["neck", "shoulder", "torso"],
        weights=1/3,
    )

    # Test utility at eye level (should be > 0)
    print("Testing utility at eye level...")
    layout_with_element_at_eye_level = get_layout_with_element_at_eye_level()
    utility_at_eye_level = get_utility(layout_with_element_at_eye_level, verbose=True)
    assert (
        utility_at_eye_level > 0
    ), "Utility should be greater than 0. Got: {}".format(
        utility_at_eye_level
    )
    print("Utility at eye level: {}".format(utility_at_eye_level))
    print()

    # Test utility at waist level
    print("Testing utility at waist level...")
    layout_with_element_at_waist_level = get_layout_with_element_at_waist_level()
    layout_with_element_at_waist_level.items[0].position.x += 0.01 # Move the UI element slightly away from the waist position
    utility_at_waist_level = get_utility(layout_with_element_at_waist_level, verbose=True)
    assert (
        utility_at_waist_level > 0
    ), "Utility should be greater than 0. Got: {}".format(
        utility_at_waist_level
    )
    print("Utility at waist level: {}".format(utility_at_waist_level))
    print()

def test_scenario_2_to_4_utility():
    """Test utility function for scenario 2 to 4."""
    print("Testing utility for scenario 2 to 4...")

    # Get utility function
    get_utility = experiments.user.get_utility_function(
        objectives=["neck", "shoulder"],
        weights=1/2,
    )

    # Test utility at eye level (should be > 0)
    print("Testing utility at eye level...")
    layout_with_element_at_eye_level = get_layout_with_element_at_eye_level()
    utility_at_eye_level = get_utility(layout_with_element_at_eye_level, verbose=True)
    assert (
        utility_at_eye_level > 0
    ), "Utility should be greater than 0. Got: {}".format(
        utility_at_eye_level
    )
    print("Utility at eye level: {}".format(utility_at_eye_level))
    print()

    # Test utility at waist level
    print("Testing utility at waist level...")
    layout_with_element_at_waist_level = get_layout_with_element_at_waist_level()
    utility_at_waist_level = get_utility(layout_with_element_at_waist_level, verbose=True)
    assert (
        utility_at_waist_level > 0
    ), "Utility should be greater than 0. Got: {}".format(
        utility_at_waist_level
    )
    print("Utility at waist level: {}".format(utility_at_waist_level))
    print()


def test_utility_functions():
    """Test various utility functions for the user models."""
    test_scenario_1_utility()
    test_scenario_2_to_4_utility()

def test_random_solver():
    """Test random solver."""
    print("Testing random solver...")

    # Get problem
    problem = experiments.problem.LayoutProblem(objectives=["neck", "shoulder"], weights=1/2)

    # Initialize random solver
    random_solver = experiments.random_solver.RandomSolver(problem=problem, seed=42)

    # Test random solver with 1 adaptation
    single_random_adaptation_layout = random_solver.get_adaptations(n_adaptations=1)[0]

    assert (
        isinstance(single_random_adaptation_layout, AUIT.networking.layout.Layout)
    ), "Adaptation should be a layout. Got: {}".format(
        type(single_random_adaptation_layout)
    )
    assert (
        problem.layout_is_valid(layout=single_random_adaptation_layout)
    ), "Adaptation should be valid. Got: {}".format(
        single_random_adaptation_layout
    )
    print("Random adaptation layout: {}".format(single_random_adaptation_layout))
    print()

    # Test random solver with 10 adaptations
    multiple_random_adaptation_layouts = random_solver.get_adaptations(n_adaptations=10)

    assert (
        isinstance(multiple_random_adaptation_layouts, list)
    ), "Adaptations should be a list. Got: {}".format(
        type(multiple_random_adaptation_layouts)
    )

    for adaptation in multiple_random_adaptation_layouts:
        assert (
            isinstance(adaptation, AUIT.networking.layout.Layout)
        ), "Adaptation should be a layout. Got: {}".format(
            type(adaptation)
        )
        assert (
            problem.layout_is_valid(layout=adaptation)
        ), "Adaptation should be valid. Got: {}".format(
            single_random_adaptation_layout
        )

    print("Random adaptation layouts: {} layouts".format(len(multiple_random_adaptation_layouts)))
    print()

    # Test seeding
    random_solver_1 = experiments.random_solver.RandomSolver(problem=problem, seed=42)
    random_solver_2 = experiments.random_solver.RandomSolver(problem=problem, seed=42)
    random_solver_3 = experiments.random_solver.RandomSolver(problem=problem, seed=43)

    assert (
        random_solver_1.get_adaptations(n_adaptations=1) == random_solver_2.get_adaptations(n_adaptations=1)
    ), "Random solvers with the same seed should return the same adaptation. Got: {} and {}".format(
        random_solver_1.get_adaptations(n_adaptations=1), random_solver_2.get_adaptations(n_adaptations=1)
    )
    assert (
        random_solver_1.get_adaptations(n_adaptations=1) != random_solver_3.get_adaptations(n_adaptations=1)
    ), "Random solvers with different seeds should return different adaptations. Got: {} and {}".format(
        random_solver_1.get_adaptations(n_adaptations=1), random_solver_3.get_adaptations(n_adaptations=1)
    )

def test_problem():
    """Test problem."""
    print("Testing problem...")

    # Test problem
    problem = experiments.problem.LayoutProblem(objectives=["neck", "shoulder"])

    # Problem should be a LayoutProblem
    assert (
        isinstance(problem, experiments.problem.LayoutProblem)
    ), "Problem should be a LayoutProblem. Got: {}".format(
        type(problem)
    )
    # Problem should have 2 objective functions
    assert (
        len(problem.objective_functions) == 2
    ), "Problem should have 2 objective functions. Got: {}".format(
        len(problem.objective_functions)
    )
    # Problem should have 7 variables
    assert (
        problem.n_var == 7
    ), "Problem should have 7 variables. Got: {}".format(
        problem.n_var
    )
    # Problem should have 7 lower bounds, the first three set to -20, the rest set to 0
    assert (
        all(problem.xl[:3] == -20) and all(problem.xl[3:] == 0)
    ), "Problem should have 7 lower bounds, the first three set to -20, the rest set to 0. Got: {}".format(
        problem.xl
    )
    # Problem should have 7 upper bounds, the first three set to 20, the rest set to 1
    assert (
        all(problem.xu[:3] == 20) and all(problem.xu[3:] == 1)
    ), "Problem should have 7 upper bounds, the first three set to 20, the rest set to 1. Got: {}".format(
        problem.xu
    )
    # Problem should have an _x_to_layout function that converts a list of 7 decision variables to a layout
    assert (
        callable(problem._x_to_layout)
    ), "Problem should have an _x_to_layout function that converts a list of 7 decision variables to a layout. Got: {}".format(
        problem._x_to_layout
    )
    # Problem should have a layout_is_valid function that checks if a layout is valid
    assert (
        callable(problem.layout_is_valid)
    ), "Problem should have a layout_is_valid function that checks if a layout is valid. Got: {}".format(
        problem.layout_is_valid
    )
    # Problem should have a is_valid function that checks if a set of decision variables is valid
    assert (
        callable(problem.is_valid)
    ), "Problem should have a is_valid function that checks if a set of decision variables is valid. Got: {}".format(
        problem.is_valid
    )
    # Problem should have a _layout_to_x function that converts a layout to a list of 7 decision variables
    assert (
        callable(problem._layout_to_x)
    ), "Problem should have a _layout_to_x function that converts a layout to a list of 7 decision variables. Got: {}".format(
        problem._layout_to_x
    )
    # Test _x_to_layout
    x = [0, 0, 0, 0, 0, 0, 0]
    layout = problem._x_to_layout(x)
    assert (
        isinstance(layout, AUIT.networking.layout.Layout)
    ), "Layout should be a Layout. Got: {}".format(
        type(layout)
    )
    assert (
        len(layout.items) == 1
    ), "Layout should have 1 element. Got: {}".format(
        (layout.items)
    )
    assert (
        layout.items[0].position.x == x[0]
    ), "Element should be at x=0. Got: {}".format(
        (layout.items[0].position.x)
    )
    assert (
        layout.items[0].position.y == x[1]
    ), "Element should be at y=0. Got: {}".format(
        (layout.items[0].position.y)
    )
    assert (
        layout.items[0].position.z == x[2]
    ), "Element should be at z=0. Got: {}".format(
        (layout.items[0].position.z)
    )
    assert (
        layout.items[0].rotation.x == x[3]
    ), "Element should be rotated x=0. Got: {}".format(
        (layout.items[0].rotation.x)
    )
    assert (
        layout.items[0].rotation.y == x[4]
    ), "Element should be rotated y=0. Got: {}".format(
        (layout.items[0].rotation.y)
    )
    assert (
        layout.items[0].rotation.z == x[5]
    ), "Element should be rotated z=0. Got: {}".format(
        (layout.items[0].rotation.z)
    )
    assert (
        layout.items[0].rotation.w == x[6]
    ), "Element should be rotated w=1. Got: {}".format(
        (layout.items[0].rotation.w)
    )
    # Test layout_is_valid
    assert (
        problem.layout_is_valid(layout=layout)
    ), "Layout should be valid. Got: {}".format(
        layout
    )
    # Test is_valid
    assert (
        problem.is_valid(x=x)
    ), "x should be valid. Got: {}".format(
        x
    )
    # Test _layout_to_x
    x_convert = problem._layout_to_x(layout=layout)
    assert (
        isinstance(x_convert, np.ndarray)
    ), "x should be a np.ndarray. Got: {}".format(
        type(x_convert)
    )
    assert (
        len(x_convert) == 7
    ), "x should have 7 elements. Got: {}".format(
        len(x_convert)
    )
    assert (
        all(x_convert == x)
    ), "x should be equal to x_convert. Got: {} and {}".format(
        x_convert, x
    )
    print("Problem: {}".format(problem))
    print()

def test_pareto_solver():
    """Test pareto solver."""
    # Create a problem
    problem = experiments.problem.LayoutProblem(
        objectives=["neck", "shoulder"]
    )
    # Create a solver
    pareto_solver = experiments.pareto_solver.ParetoSolver(problem=problem, pop=100, n_gen=100, seed=42)
    # Test getting a single adaptation solution
    pareto_optimal_adaptation = pareto_solver.get_adaptations(decomposition="full", verbose=True)[0]
    assert (
        isinstance(pareto_optimal_adaptation, AUIT.networking.layout.Layout)
    ), "Adaptation should be a layout. Got: {}".format(
        type(pareto_optimal_adaptation)
    )
    assert (
        problem.layout_is_valid(layout=pareto_optimal_adaptation)
    ), "Adaptation should be valid. Got: {}".format(
        pareto_optimal_adaptation
    )
    print("Pareto optimal adaptation layout (equal weight AASF): {}".format(pareto_optimal_adaptation))
    print()

    # Test getting multiple adaptation solutions (AASF)
    pareto_optimal_adaptations_aasf = pareto_solver.get_adaptations(decomposition="aasf", verbose=True)
    assert (
        isinstance(pareto_optimal_adaptations_aasf, list)
    ), "Adaptations should be a list. Got: {}".format(
        type(pareto_optimal_adaptations_aasf)
    )
    assert (
        len(pareto_optimal_adaptations_aasf) == problem.n_obj + 1
    ), "Adaptations should have n_obj+1 elements. Got: {}".format(
        len(pareto_optimal_adaptations_aasf)
    )
    for adaptation in pareto_optimal_adaptations_aasf:
        assert (
            isinstance(adaptation, AUIT.networking.layout.Layout)
        ), "Adaptation should be a layout. Got: {}".format(
            type(adaptation)
        )
        assert (
            problem.layout_is_valid(layout=adaptation)
        ), "Adaptation should be valid. Got: {}".format(
            adaptation
        )
    print("Pareto optimal adaptation layouts (n_obj+1 AASF): {}".format(pareto_optimal_adaptations_aasf))
    print()

    # Test getting multiple adaptation solutions (WS)
    pareto_optimal_adaptation_ws = pareto_solver.get_adaptations(decomposition="ws", verbose=True)[0]
    assert (
        isinstance(pareto_optimal_adaptation_ws, AUIT.networking.layout.Layout)
    ), "Adaptation should be a layout. Got: {}".format(
        type(pareto_optimal_adaptation_ws)
    )
    assert (
        problem.layout_is_valid(layout=pareto_optimal_adaptation_ws)
    ), "Adaptation should be valid. Got: {}".format(
        pareto_optimal_adaptation_ws
    )
    print("Pareto optimal adaptation layout (WS): {}".format(pareto_optimal_adaptation_ws))
    print()


    # Test getting multiple adaptation solutions (whole Pareto front)
    pareto_front_adaptations = pareto_solver.get_adaptations(decomposition=None, verbose=True)
    assert (
        isinstance(pareto_front_adaptations, list)
    ), "Adaptations should be a list. Got: {}".format(
        type(pareto_front_adaptations)
    )
    for adaptation in pareto_front_adaptations:
        assert (
            isinstance(adaptation, AUIT.networking.layout.Layout)
        ), "Adaptation should be a layout. Got: {}".format(
            type(adaptation)
        )
        assert (
            problem.layout_is_valid(layout=adaptation)
        ), "Adaptation should be valid. Got: {}".format(
            adaptation
        )
    print("Pareto front adaptation layouts: {} layouts".format(len(pareto_front_adaptations)))
    print()

def test_weighted_sum_solver():
    """Test weighted sum solver."""
    # Create a problem
    problem = experiments.problem.LayoutProblem(
        objectives=["neck", "shoulder"]
    )
    # Create a solver
    weighted_sum_solver = experiments.weighted_sum_solver.WeightedSumSolver(problem=problem, weights=1/problem.n_obj, pop=100, n_gen=100, seed=42)
    # Test getting a single adaptation solution
    weighted_sum_adaptation = weighted_sum_solver.get_adaptations(verbose=True)[0]
    assert (
        isinstance(weighted_sum_adaptation, AUIT.networking.layout.Layout)
    ), "Adaptation should be a layout. Got: {}".format(
        type(weighted_sum_adaptation)
    )
    assert (
        problem.layout_is_valid(layout=weighted_sum_adaptation)
    ), "Adaptation should be valid. Got: {}".format(
        weighted_sum_adaptation
    )
    print("Weighted sum adaptation layout: {}".format(weighted_sum_adaptation))
    print()



def test_evaluation():
    """Test evaluations."""
    test_utility_functions()
    test_random_solver()
    test_problem()
    test_weighted_sum_solver()
    test_pareto_solver()


def main():
    """Main function."""
    test_evaluation()


if __name__ == "__main__":
    main()
