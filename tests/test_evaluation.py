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
import experiments.single_objectives_solver
import experiments.problem
import experiments.simulate
import AUIT
import numpy as np
import pandas as pd
from typing import Callable
from pymoo.util.ref_dirs import get_reference_directions

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

def test_utility_function_at_various_positions(get_utility: Callable):
    """Test a utility function at various positions."""
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

def test_utility_population():
    """Test utility population function."""
    print("Testing utility population...")

    # Set random seed
    seed = 42

    # Get 1 utility function with specified weights
    get_utility = experiments.user.get_utility_functions(
        objectives=["neck", "shoulder"],
        weights=1/2,
        n_functions=1,
        seed=seed,
    )[0]

    # Test utility at various positions
    test_utility_function_at_various_positions(get_utility)

    # Get 1 utility function with random weights
    get_utility = experiments.user.get_utility_functions(
        objectives=["neck", "shoulder"],
        weights=None,
        n_functions=1,
        seed=seed,
    )[0]

    # Test utility at various positions
    test_utility_function_at_various_positions(get_utility)

    # Get 10 utility functions with random weights
    get_utilities = experiments.user.get_utility_functions(
        objectives=["neck", "shoulder"],
        weights=None,
        n_functions=10,
        seed=seed,
    )

    # Test utility at various positions
    for get_utility in get_utilities:
        test_utility_function_at_various_positions(get_utility)


def test_utility_functions():
    """Test various utility functions for the user models."""
    test_utility_population()
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
    # Problem should have 7 lower bounds
    # The first and third (x and z bounds) should be set to -3, the second (y bound) should be set to -2, the rest set to 0
    assert (
        problem.xl[0] == -3 and problem.xl[1] == -2 and problem.xl[2] == -3 and all(problem.xl[3:] == 0)
    ), "Problem should have 7 lower bounds, the first and third set to -3, the second set to -2, the rest set to 0. Got: {}".format(
        problem.xl
    )
    # Problem should have 7 upper bounds
    # The first and third (x and z bounds) should be set to 3, the second (y bound) should be set to 2, the rest set to 1
    assert (
        problem.xu[0] == 3 and problem.xu[1] == 2 and problem.xu[2] == 3 and all(problem.xu[3:] == 1)
    ), "Problem should have 7 upper bounds, the first and third set to 3, the second set to 2, the rest set to 1. Got: {}".format(
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
        len(pareto_optimal_adaptations_aasf) == 10
    ), "Adaptations should have 10 elements. Got: {}".format(
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
    print("Pareto optimal adaptation layouts (10 AASF): {}".format(pareto_optimal_adaptations_aasf))
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
    assert (
        len(pareto_front_adaptations) > 0
    ), "Adaptations should have more than 0 elements. Got: {}".format(
        len(pareto_front_adaptations)
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

    def test_solution(layout):
        """Test solution."""
        assert (
            isinstance(layout, AUIT.networking.layout.Layout)
        ), "Adaptation should be a layout. Got: {}".format(
            type(layout)
        )
        assert (
            problem.layout_is_valid(layout=layout)
        ), "Adaptation should be valid. Got: {}".format(
            layout
        )
        print("Weighted sum adaptation layout: {}".format(layout))
        print()


    # Create a solver
    weighted_sum_solver = experiments.weighted_sum_solver.WeightedSumSolver(problem=problem, weights=1/problem.n_obj, pop=100, n_gen=100, seed=42)
    # Test getting a single adaptation solution
    weighted_sum_adaptation = weighted_sum_solver.get_adaptations(max_n_proposals=1, verbose=True)[0]
    test_solution(weighted_sum_adaptation)

    # Test Nelder Mead algorithm for weighted sum solver
    weighted_sum_solver = experiments.weighted_sum_solver.WeightedSumSolver(problem=problem, weights=1/problem.n_obj, pop=100, n_gen=100, seed=42, algo="nm")
    weighted_sum_adaptation_nm = weighted_sum_solver.get_adaptations(max_n_proposals=1, verbose=True)[0]
    test_solution(weighted_sum_adaptation_nm)

    # Test NSGA-III single-objective algorithm for weighted sum solver
    weighted_sum_solver = experiments.weighted_sum_solver.WeightedSumSolver(problem=problem, weights=1/problem.n_obj, pop=100, n_gen=100, seed=42, algo="nsga3")
    weighted_sum_adaptation_nsga3 = weighted_sum_solver.get_adaptations(max_n_proposals=1, verbose=True)[0]
    test_solution(weighted_sum_adaptation_nsga3)

    # Test getting multiple adaptation solutions using NSGA-III
    weighted_sum_adaptations_nsga3 = weighted_sum_solver.get_adaptations(max_n_proposals=10, verbose=True)
    assert (
        isinstance(weighted_sum_adaptations_nsga3, list)
    ), "Adaptations should be a list. Got: {}".format(
        type(weighted_sum_adaptations_nsga3)
    )
    assert (
        len(weighted_sum_adaptations_nsga3) > 1 and len(weighted_sum_adaptations_nsga3) <= 10
    ), "Adaptations should have between 2 and 10 elements. Got: {}".format(
        len(weighted_sum_adaptations_nsga3)
    )
    assert (
        len(weighted_sum_adaptations_nsga3) == len(set([str(adaptation) for adaptation in weighted_sum_adaptations_nsga3]))
    ), "Adaptations should be unique. Got: {}".format(
        weighted_sum_adaptations_nsga3
    )
    for adaptation in weighted_sum_adaptations_nsga3:
        test_solution(adaptation)


def test_multiple_single_objectives_solver():
    """Test multiple single objectives solver."""
    # Create a problem
    problem = experiments.problem.LayoutProblem(
        objectives=["neck", "shoulder"]
    )
    # Create a solver
    multiple_single_objectives_solver = experiments.single_objectives_solver.SingleObjectivesSolver(problem=problem, pop=100, n_gen=100, seed=42)
    # Test getting the adaptations representing the single objective solutions
    single_objectives_adaptations = multiple_single_objectives_solver.get_adaptations(verbose=True)
    assert (
        isinstance(single_objectives_adaptations, list)
    ), "Adaptations should be a list. Got: {}".format(
        type(single_objectives_adaptations)
    )
    assert (
        len(single_objectives_adaptations) == problem.n_obj
    ), "Adaptations should have n_obj elements. Got: {}".format(
        len(single_objectives_adaptations)
    )
    for adaptation in single_objectives_adaptations:
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
    print("Single objective adaptation layouts: {}".format(single_objectives_adaptations))
    print()

def test_simulations():
    """Test simulation functions used in the Jupyter notebook."""
    print("Testing simulations...")

    N_UTILITY_FUNCTIONS = 20
    N_RUNS = 3
    N_PROPOSALS = [1, 10]
    SOLVERS = ["Ours", "WS"]

    # Get utility functions
    SCENARIO_1_PREFERENCE_CRITERIA = ["neck", "shoulder", "torso"]
    utility_functions = experiments.user.get_utility_functions_for_different_seeds(
        SCENARIO_1_PREFERENCE_CRITERIA, n_functions=N_UTILITY_FUNCTIONS, seed=111
    )

    # Get MOO problem
    SCENARIO_1_OBJECTIVES = ["neck", "shoulder"]
    problem = experiments.problem.LayoutProblem(
        objectives=SCENARIO_1_OBJECTIVES
    )

    # Get runtimes and results
    runtimes, results = experiments.simulate.get_runtimes_and_results_dfs(
        problem=problem,
        scenario="TEST",
        utility_functions=utility_functions,
        n_runs=N_RUNS,
        seed=42,
    )

    # Assert that the runtimes are correct
    # Check that runtimes is a pd.DataFrame with the correct columns
    assert (
        isinstance(runtimes, pd.DataFrame)
    ), "Runtimes should be a pd.DataFrame. Got: {}".format(
        type(runtimes)
    )
    EXPECTED_COLUMNS = [
        "run_id",
        "scenario",
        "solver",
        "n_proposals",
        "run_iter",
        "seed",
        "start_time",
        "end_time",
        "runtime",
    ]
    assert (
        set(runtimes.columns) == set(EXPECTED_COLUMNS)
    ), "Runtimes should have columns: {}. Got: {}".format(
        EXPECTED_COLUMNS, runtimes.columns
    )

    # Check that the number of rows is correct
    assert (
        runtimes.shape[0] == N_RUNS * len(N_PROPOSALS) * len(SOLVERS)
    ), "Runtimes should have {} rows. Got: {}".format(
        N_RUNS * len(N_PROPOSALS) * len(SOLVERS), runtimes.shape[0]
    )

    # Check that the scenario is correct
    assert (
        runtimes["scenario"].unique()[0] == "TEST"
    ), "Runtimes should have scenario TEST. Got: {}".format(
        runtimes["scenario"].unique()[0]
    )

    # Check that the solver is correct
    assert (
        set(runtimes["solver"].unique()) == set(SOLVERS)
    ), "Runtimes should have solver {}. Got: {}".format(
        SOLVERS, runtimes["solver"].unique()
    )

    # Check that the number of proposals is correct
    assert (
        set(runtimes["n_proposals"].unique()) == set(N_PROPOSALS)
    ), "Runtimes should have n_proposals {}. Got: {}".format(
        N_PROPOSALS, runtimes["n_proposals"].unique()[0]
    )

    # Check that the run_iter is correct (ranging from 1 to 10)
    assert (
        set(runtimes["run_iter"].unique()) == set(range(1, N_RUNS + 1))
    ), "Runtimes should have run_iter ranging from 1 to {}. Got: {}".format(
        N_RUNS, runtimes["run_iter"].unique()
    )

    # Check that the seed is correct (no duplicates across runs)
    assert (
        len(runtimes["seed"].unique()) == N_RUNS
    ), "Runs should have unique seeds. Got: {}".format(
        runtimes["seed"].unique()
    )

    # Assert that the results are correct
    # Check that results is a pd.DataFrame with the correct columns
    assert (
        isinstance(results, pd.DataFrame)
    ), "Results should be a pd.DataFrame. Got: {}".format(
        type(results)
    )
    EXPECTED_COLUMNS = [
        "run_id",
        "utility_id",
        "adaptation_id",
        "utility",
    ]
    assert (
        set(results.columns) == set(EXPECTED_COLUMNS)
    ), "Results should have columns: {}. Got: {}".format(
        EXPECTED_COLUMNS, results.columns
    )

    # Check that the number of rows is correct
    assert (
        results.shape[0] == N_RUNS * N_UTILITY_FUNCTIONS * sum(N_PROPOSALS) * len(SOLVERS)
    ), "Results should have {} rows. Got: {}".format(
        N_RUNS * N_UTILITY_FUNCTIONS * sum(N_PROPOSALS) * len(SOLVERS), results.shape[0]
    )

    # Test that you can get the expected utility for a given set of preference criteria
    test_get_expected_utility()


def test_get_expected_utility():
    """Test that you can get the expected utility for a given set of preference criteria."""
    # Get utility functions
    PREFERENCE_CRITERIA = ["neck", "shoulder", "torso"]
    utility_functions = experiments.user.get_utility_functions(PREFERENCE_CRITERIA, n_functions=20, seed=42)
    expected_utility = experiments.simulate.get_expected_utility(
        PREFERENCE_CRITERIA,
        utility_functions,
        n_trials=1000,
        seed=42,
    )
    assert (
        isinstance(expected_utility, float)
    ), "Expected utility should be a float. Got: {}".format(
        type(expected_utility)
    )
    assert (
        expected_utility > 0
    ), "Expected utility should be positive. Got: {}".format(
        expected_utility
    )



def test_riesz():
    """Test Riesz s-Energy reference points to generate well-spaced weights."""
    print("Testing Riesz s-Energy reference points...")

    N_OBJ = [2, 3, 4]
    N_REF_POINTS = [1, 5, 10]

    def test_shape(ref_dirs, n_obj, n_ref_points):
        """Assert that the reference points have the right shape."""
        # Check that ref_dirs is a np.ndarray with the correct shape
        assert (
            isinstance(ref_dirs, np.ndarray)
        ), "Reference points should be a np.ndarray. Got: {}".format(
            type(ref_dirs)
        )

        assert (
            ref_dirs.shape == (n_ref_points, n_obj)
        ), "Reference points should have shape: ({}, {}). Got: {}".format(
            n_ref_points, n_obj, ref_dirs.shape
        )

    def test_well_spaced(ref_dirs):
        """Check that the reference points are well-spaced."""
        # Minimum distance between any two neighboring reference points should be similar across
        # all neighboring reference points
        # Compute pairwise distances using numpy
        pairwise_distances = np.linalg.norm(
            ref_dirs[:, None] - ref_dirs[None, :], axis=-1
        )
        # Set diagonal to infinity
        np.fill_diagonal(pairwise_distances, np.inf)
        # Get minimum distance between any two neighboring reference points
        min_distances = np.min(pairwise_distances, axis=1)
        # Check that the minimum distance between any two neighboring reference points is similar
        # across all neighboring reference points
        assert (
            np.allclose(min_distances, min_distances[0], rtol=5e-2, atol=5e-2)
        ), "Reference points should be well-spaced. Got: {}".format(min_distances)

    for n_obj in N_OBJ:
        for n_ref_points in N_REF_POINTS:
            ref_dirs = get_reference_directions(
                "energy", n_obj, n_ref_points, seed=42
            )
            test_shape(ref_dirs, n_obj, n_ref_points)
            test_well_spaced(ref_dirs)


def test_evaluation():
    """Test evaluations."""
    # test_riesz()
    test_simulations()
    # test_utility_functions()
    # test_random_solver()
    # test_problem()
    # test_multiple_single_objectives_solver()
    # test_weighted_sum_solver()
    # test_pareto_solver()


def main():
    """Main function."""
    test_evaluation()


if __name__ == "__main__":
    main()
