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
import experiments.sensitivity
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

    test_utility_function_at_various_positions(get_utility)

    

def test_scenario_2_to_4_utility():
    """Test utility function for scenario 2 to 4."""
    print("Testing utility for scenario 2 to 4...")

    # Get utility function
    get_utility = experiments.user.get_utility_function(
        objectives=["neck", "shoulder"],
        weights=1/2,
    )

    test_utility_function_at_various_positions(get_utility)


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


def test_shoulder_utility_function():
    """Test shoulder utility function."""
    print("Testing shoulder utility function...")

    # Get shoulder utility function
    get_shoulder_utility = experiments.user.get_utility_function(objectives=["shoulder"], weights=1.0)

    # Test utility at various positions
    test_utility_function_at_various_positions(get_shoulder_utility)

    # Test utility at eye level (cost should be greater than 0, utility should be less than 1)
    print("Testing utility at eye level...")
    layout_with_element_at_eye_level = get_layout_with_element_at_eye_level()
    utility_at_eye_level = get_shoulder_utility(layout_with_element_at_eye_level, verbose=True)
    cost_at_eye_level = 1 - utility_at_eye_level
    assert (
        utility_at_eye_level < 1
    ), "Utility should be less than 1. Got: {}".format(
        utility_at_eye_level
    )
    print("Utility at eye level: {}".format(utility_at_eye_level))
    print()

    # Test cost at eye level (cost should be greater than 0, utility should be less than 1)
    print("Testing cost at eye level...")
    assert (
        cost_at_eye_level > 0
    ), "Cost should be greater than 0. Got: {}".format(
        cost_at_eye_level
    )

    # Test utility at waist level (should approach 1, cost should approach 0)
    print("Testing utility at waist level...")
    layout_with_element_at_waist_level = get_layout_with_element_at_waist_level()
    utility_at_waist_level = get_shoulder_utility(layout_with_element_at_waist_level, verbose=True)
    cost_at_waist_level = 1 - utility_at_waist_level
    assert (
        utility_at_waist_level > 0.9
    ), "Utility should be greater than 0.9. Got: {}".format(
        utility_at_waist_level
    )
    print("Utility at waist level: {}".format(utility_at_waist_level))
    print()

    # Test cost at waist level (should approach 1, cost should approach 0)
    print("Testing cost at waist level...")
    assert (
        cost_at_waist_level < 0.1
    ), "Cost should be less than 0.9. Got: {}".format(
        cost_at_waist_level
    )


def test_utility_functions():
    """Test various utility functions for the user models."""
    test_shoulder_utility_function()
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
    SCENARIO_1_PREFERENCE_CRITERIA = ["neck", "shoulder_exp"]
    utility_functions = experiments.user.get_utility_functions_for_different_seeds(
        SCENARIO_1_PREFERENCE_CRITERIA, n_functions=N_UTILITY_FUNCTIONS, seed=111
    )

    # Get MOO problem
    SCENARIO_1_OBJECTIVES = ["neck", "shoulder_exp"]
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

def test_get_clutter():
    """Test function to get objects cluttered in the user's environment."""
    number_of_objects_params = [1, 2, 4, 8, 16, 32, 64, 128]
    INTERACTION_VOLUME_RADIUS = 2
    for number_of_objects in number_of_objects_params:
        object_locations = experiments.sensitivity.get_object_locations(
            number_of_objects=number_of_objects,
            interaction_volume_radius=INTERACTION_VOLUME_RADIUS,
            seed=42,
        )
        assert (
            len(object_locations) == number_of_objects
        ), "Object locations should have {} objects. Got: {}".format(
            number_of_objects, len(object_locations)
        )
        assert (
            isinstance(object_locations, list)
        ), "Object locations should be a list. Got: {}".format(
            type(object_locations)
        )
        for object_location in object_locations:
            assert (
                isinstance(object_location, AUIT.networking.element.Position)
            ), "Object location should be a Position. Got: {}".format(
                type(object_location)
            )
            assert (
                object_location.x >= -INTERACTION_VOLUME_RADIUS and object_location.x <= INTERACTION_VOLUME_RADIUS
            ), "Object location should be within the interaction volume. Got: {}".format(
                object_location
            )
            assert (
                object_location.y >= -INTERACTION_VOLUME_RADIUS and object_location.y <= INTERACTION_VOLUME_RADIUS
            ), "Object location should be within the interaction volume. Got: {}".format(
                object_location
            )
            assert (
                object_location.z >= -INTERACTION_VOLUME_RADIUS and object_location.z <= INTERACTION_VOLUME_RADIUS
            ), "Object location should be within the interaction volume. Got: {}".format(
                object_location
            )

def test_get_association_strength():
    """Test function to get positive and negative association scores based on an input
    variance for the joint association distribution."""
    association_score_variance_params = [0.1, 0.5, 1, 2, 5, 10]
    NUMBER_OF_OBJECTS = 1000
    all_positive_association_scores = []
    all_negative_association_scores = []
    for association_score_variance in association_score_variance_params:
        association_scores = experiments.sensitivity.get_association_scores(
            number_of_objects=NUMBER_OF_OBJECTS,
            association_score_variance=association_score_variance,
            seed=42,
        )
        assert (
            len(association_scores) == NUMBER_OF_OBJECTS
        ), "Association scores should have {} objects. Got: {}".format(
            NUMBER_OF_OBJECTS, len(association_scores)
        )
        assert (
            isinstance(association_scores, list)
        ), "Association scores should be a list. Got: {}".format(
            type(association_scores)
        )
        for association_score in association_scores:
            assert (
                isinstance(association_score, dict)
            ), "Association score should be a dict. Got: {}".format(
                type(association_score)
            )
            assert (
                "positive_association_score" in association_score
            ), "Association score should have a positive_association_score key. Got: {}".format(
                association_score
            )
            assert (
                "negative_association_score" in association_score
            ), "Association score should have a negative_association_score key. Got: {}".format(
                association_score
            )
            assert (
                isinstance(association_score["positive_association_score"], float)
            ), "Association score should be a float. Got: {}".format(
                type(association_score["positive_association_score"])
            )
            assert (
                isinstance(association_score["negative_association_score"], float)
            ), "Association score should be a float. Got: {}".format(
                type(association_score["negative_association_score"])
            )
            assert (
                association_score["positive_association_score"] >= 0 and association_score["positive_association_score"] <= 1
            ), "Association score should be between 0 and 1. Got: {}".format(
                association_score["positive_association_score"]
            )
            assert (
                association_score["negative_association_score"] >= 0 and association_score["negative_association_score"] <= 1
            ), "Association score should be between 0 and 1. Got: {}".format(
                association_score["negative_association_score"]
            )
        # Check that the positive and negative association scores are correlated
        positive_association_scores = [association_score["positive_association_score"] for association_score in association_scores]
        negative_association_scores = [association_score["negative_association_score"] for association_score in association_scores]
        all_positive_association_scores.append(positive_association_scores)
        all_negative_association_scores.append(negative_association_scores)
        assert (
            np.corrcoef(positive_association_scores, negative_association_scores)[0, 1] > 0
        ), "Positive and negative association scores should be positively correlated. Got: {}".format(
            np.corrcoef(positive_association_scores, negative_association_scores)[0, 1]
        )
        # Check that the variance of the positive and negative association scores is similar
        assert (
            np.var(positive_association_scores) / np.var(negative_association_scores) > 0.8 and np.var(positive_association_scores) / np.var(negative_association_scores) < 1.2
        ), "Variance of positive and negative association scores should be similar. Got: {}".format(
            np.var(positive_association_scores) / np.var(negative_association_scores)
        )
        # Check that the mean of the positive and negative association scores is similar
        assert (
            np.mean(positive_association_scores) / np.mean(negative_association_scores) > 0.8 and np.mean(positive_association_scores) / np.mean(negative_association_scores) < 1.2
        ), "Mean of positive and negative association scores should be similar. Got: {}".format(
            np.mean(positive_association_scores) / np.mean(negative_association_scores)
        )
        # Check that the positive and negative association scores are not too close to 0 or 1
        assert (
            np.mean(positive_association_scores) > 0.1 and np.mean(positive_association_scores) < 0.9
        ), "Mean of positive association scores should be between 0.1 and 0.9. Got: {}".format(
            np.mean(positive_association_scores)
        )
        assert (
            np.mean(negative_association_scores) > 0.1 and np.mean(negative_association_scores) < 0.9
        ), "Mean of negative association scores should be between 0.1 and 0.9. Got: {}".format(
            np.mean(negative_association_scores)
        )
    # Check that the variance of the positive and negative association scores is correlated
    # with the input variance
    assert (
        np.corrcoef(association_score_variance_params, [np.var(association_scores) for association_scores in all_positive_association_scores])[0, 1] > 0.6
    ), "Variance of positive association scores should be correlated with input variance. Got: {}".format(
        np.corrcoef(association_score_variance_params, [np.var(association_scores) for association_scores in all_positive_association_scores])[0, 1]
    )
    assert (
        np.corrcoef(association_score_variance_params, [np.var(association_scores) for association_scores in all_negative_association_scores])[0, 1] > 0.6
    ), "Variance of negative association scores should be correlated with input variance. Got: {}".format(
        np.corrcoef(association_score_variance_params, [np.var(association_scores) for association_scores in all_negative_association_scores])[0, 1]
    )

def test_get_clutter_association_grid():
    """Test function to create a grid (i.e., a matrix with lists of objects) based on the
    number of object and association strength variance inputs."""
    # Make a logarithmic grid of number of objects and association strength variance
    number_of_objects_params = np.logspace(base=2, start=0, stop=6, num=7, dtype=int) # 1 to 64
    association_score_variance_params = np.logspace(base=0.5, start=3, stop=-3, num=7) # 0.125 to 8
    association_dicts = experiments.sensitivity.get_clutter_association_grid(
        number_of_objects_params=number_of_objects_params,
        association_score_variance_params=association_score_variance_params,
        seed=42,
    )
    assert (
        isinstance(association_dicts, list)
    ), "Association dicts should be a list. Got: {}".format(
        type(association_dicts)
    )
    assert (
        len(association_dicts) == len(number_of_objects_params) * len(association_score_variance_params)
    ), "Association dicts should have {} elements. Got: {}".format(
        len(number_of_objects_params) * len(association_score_variance_params), len(association_dicts)
    )
    for association_dict in association_dicts:
        assert (
            isinstance(association_dict, dict)
        ), "Association dict should be a dict. Got: {}".format(
            type(association_dict)
        )
        assert (
            "objects" in association_dict
        ), "Association dict should have a objects key. Got: {}".format(
            association_dict
        )
        assert (
            isinstance(association_dict["objects"], list)
        ), "Association dict should have a list of objects. Got: {}".format(
            type(association_dict["objects"])
        )
        assert (
            "number_of_objects" in association_dict
        ), "Association dict should have a number_of_objects key. Got: {}".format(
            association_dict
        )
        assert (
            isinstance(association_dict["number_of_objects"], int)
        ), "Association dict should have an int for number_of_objects. Got: {}".format(
            type(association_dict["number_of_objects"])
        )
        assert (
            "association_score_variance" in association_dict
        ), "Association dict should have a association_score_variance key. Got: {}".format(
            association_dict
        )
        assert (
            isinstance(association_dict["association_score_variance"], float)
        ), "Association dict should have a float for association_score_variance. Got: {}".format(
            type(association_dict["association_score_variance"])
        )
        assert (
            association_dict["number_of_objects"] in number_of_objects_params
        ), "Association dict should have a number_of_objects in number_of_objects_params. Got: {}".format(
            association_dict["number_of_objects"]
        )
        assert (
            association_dict["association_score_variance"] in association_score_variance_params
        ), "Association dict should have a association_score_variance in association_score_variance_params. Got: {}".format(
            association_dict["association_score_variance"]
        )
        assert (
            len(association_dict["objects"]) == association_dict["number_of_objects"]
        ), "Association dict should have {} objects. Got: {}".format(
            association_dict["number_of_objects"], len(association_dict["objects"])
        )

def test_get_semantic_costs_for_grid():
    """Test the function to get the associated minimum semantic cost for each cell in a grid
    with a set number of objects and association strength variance."""
    number_of_objects_params = np.logspace(base=2, start=0, stop=6, num=7, dtype=int) # 1 to 64
    association_score_variance_params = np.logspace(base=2, start=1, stop=7, num=7)/100. # 0.02 to 1.28
    association_dicts = experiments.sensitivity.get_clutter_association_grid(
        number_of_objects_params=number_of_objects_params,
        association_score_variance_params=association_score_variance_params,
        seed=42,
    )
    semantic_costs = experiments.sensitivity.get_minimum_semantic_costs_for_grid(
        association_dicts=association_dicts,
        seed=42,
    )
    assert (
        isinstance(semantic_costs, list)
    ), "Semantic costs should be a list. Got: {}".format(
        type(semantic_costs)
    )
    assert (
        len(semantic_costs) == len(number_of_objects_params) * len(association_score_variance_params)
    ), "Semantic costs should have {} elements. Got: {}".format(
        len(number_of_objects_params) * len(association_score_variance_params), len(semantic_costs)
    )
    # Check that not all semantic costs are the same
    assert (
        len(set(semantic_costs)) > 1
    ), "Semantic costs should not all be the same. Got: {}".format(
        semantic_costs
    )
    for semantic_cost in semantic_costs:
        assert (
            isinstance(semantic_cost, float)
        ), "Semantic cost should be a float. Got: {}".format(
            type(semantic_cost)
        )

def test_get_semantic_costs_grid_dataframe():
    """Test the function to get a DataFrame with the semantic costs for each cell in a grid
    with a set number of objects and association strength variance."""
    number_of_objects_params = np.logspace(base=2, start=0, stop=6, num=7, dtype=int) # 1 to 64
    association_score_variance_params = np.logspace(base=2, start=1, stop=7, num=7)/100. # 0.02 to 1.28
    association_dicts = experiments.sensitivity.get_clutter_association_grid(
        number_of_objects_params=number_of_objects_params,
        association_score_variance_params=association_score_variance_params,
        seed=42,
    )
    semantic_costs = experiments.sensitivity.get_minimum_semantic_costs_for_grid(
        association_dicts=association_dicts,
        seed=42,
    )
    semantic_costs_grid_df = experiments.sensitivity.get_semantic_costs_grid_dataframe(
        association_dicts=association_dicts,
        semantic_costs=semantic_costs,
    )
    assert (
        isinstance(semantic_costs_grid_df, pd.DataFrame)
    ), "Semantic costs grid should be a pd.DataFrame. Got: {}".format(
        type(semantic_costs_grid_df)
    )
    assert (
        semantic_costs_grid_df.shape[0] == len(number_of_objects_params) * len(association_score_variance_params)
    ), "Semantic costs grid should have {} rows. Got: {}".format(
        len(number_of_objects_params) * len(association_score_variance_params), semantic_costs_grid_df.shape[0]
    )
    assert (
        semantic_costs_grid_df.shape[1] == 3
    ), "Semantic costs grid should have 3 columns. Got: {}".format(
        semantic_costs_grid_df.shape[1]
    )
    assert (
        set(semantic_costs_grid_df.columns) == set(["number_of_objects", "association_score_variance", "minimum_semantic_cost"])
    ), "Semantic costs grid should have columns: {}. Got: {}".format(
        ["number_of_objects", "association_score_variance", "minimum_semantic_cost"], semantic_costs_grid_df.columns
    )
    for index, row in semantic_costs_grid_df.iterrows():
        assert (
            row["number_of_objects"] in number_of_objects_params
        ), "Semantic costs grid should have a number_of_objects in number_of_objects_params. Got: {}".format(
            row["number_of_objects"]
        )
        assert (
            row["association_score_variance"] in association_score_variance_params
        ), "Semantic costs grid should have a association_score_variance in association_score_variance_params. Got: {}".format(
            row["association_score_variance"]
        )
        assert (
            row["minimum_semantic_cost"] in semantic_costs
        ), "Semantic costs grid should have a minimum_semantic_cost in semantic_costs. Got: {}".format(
            row["minimum_semantic_cost"]
        )

def test_sensitivity_of_semantic_cost():
    """Test the sensitivity of the semantic cost objective on clutter (i.e., density of objects
    in the user's environment) and on association strength (i.e., variance of positive and negative
    association scores)."""
    test_get_clutter()
    test_get_association_strength()
    test_get_clutter_association_grid()
    test_get_semantic_costs_for_grid()
    test_get_semantic_costs_grid_dataframe()

def test_sensitivity_of_max_utility_with_semantic_cost():
    """Test the sensitivity of the maximum utility for a simulated population of users with
    underspecified preferences which include the semantic cost to clutter and association strength
    for both the weighted sum solver and the Pareto solver."""
    pass

def test_sensitivity_analysis():
    """Test the required functions for sensitivity analysis of the semantic cost in AUIT."""
    test_sensitivity_of_semantic_cost()
    test_sensitivity_of_max_utility_with_semantic_cost()


def test_evaluation():
    """Test evaluations."""
    test_sensitivity_analysis()
    test_simulations()
    test_utility_functions()
    test_riesz()
    test_random_solver()
    test_problem()
    test_multiple_single_objectives_solver()
    test_weighted_sum_solver()
    test_pareto_solver()


def main():
    """Main function."""
    test_evaluation()


if __name__ == "__main__":
    main()
