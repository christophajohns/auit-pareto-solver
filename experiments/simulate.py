"""Functions to create statistics for the results of the experiments."""

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

from .pareto_solver import ParetoSolver
from .problem import LayoutProblem
from .weighted_sum_solver import WeightedSumSolver
from .random_solver import RandomSolver
import numpy as np
import datetime as dt
import pandas as pd
import uuid
from .config import OBJECTIVE, UTILITY_FUNCTION

def get_runtimes_and_results_dfs(problem, scenario, utility_functions, n_runs=10, seed=42):
    """Get runtimes and results DataFrames for n_runs of the Pareto solver
    with different seeds."""
    runtimes, results = get_utilities_for_pareto_solver(problem, scenario, utility_functions, n_runs=n_runs, seed=seed)
    runtimes_flat = [
        {
            "run_id": run_id,
            "scenario": scenario,
            "solver": run["solver"],
            "n_proposals": run["n_proposals"],
            "run_iter": run["run_iter"],
            "seed": run["seed"],
            "start_time": run["start_time"],
            "end_time": run["end_time"],
            "runtime": run["runtime"],
        }
        for run_id, run in runtimes.items()
    ]
    results_flat = [
        {
            "run_id": run_id,
            "utility_id": utility_id,
            "adaptation_id": adaptation_id,
            "utility": utility,
        }
        for run_id, run in results.items()
        for utility_id, utilities_dict in run.items()
        for adaptation_id, utility in utilities_dict.items()
    ]
    runtimes_df = pd.DataFrame(runtimes_flat)
    results_df = pd.DataFrame(results_flat)
    return runtimes_df, results_df

def get_utilities_for_pareto_solver(problem, scenario, utility_functions, n_runs=10, seed=42):
    """Get utilities for adaptations generaed via n_runs of the Pareto solver
    with different seeds."""
    runtimes = {}
    results = {}
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**8, size=n_runs)
    solver_labels = ["WS", "Ours"]

    def optimize(solver_label, get_adaptations, n_proposals):
        for run_iter, seed in enumerate(seeds):
            run_id = str(uuid.uuid4())
            runtimes[run_id] = {
                "run_id": run_id,
                "scenario": scenario,
                "solver": solver_label,
                "n_proposals": n_proposals,
                "run_iter": run_iter+1,
                "seed": seed,
            }
            results[run_id] = {}
            runtimes[run_id]["start_time"] = dt.datetime.now()
            optimal_adaptations = get_adaptations()
            adaptations = {
                str(uuid.uuid4()): adaptation
                for adaptation in optimal_adaptations
            }
            runtimes[run_id]["end_time"] = dt.datetime.now()
            runtimes[run_id]["runtime"] = (runtimes[run_id]["end_time"] - runtimes[run_id]["start_time"]).total_seconds()
            utilities = get_utilities_for_adaptations(adaptations, utility_functions)
            for utility_id, utility in utilities.items():
                results[run_id][utility_id] = utility.copy()

    for solver in solver_labels:
        if solver == "WS":
            for n_proposals in [1, 10]:
                ws_solver = WeightedSumSolver(problem, algo="nsga3", pop=100, n_gen=100, seed=seed)
                optimize(solver, lambda: ws_solver.get_adaptations(n_proposals), n_proposals)
        else:
            decompositions = ["full", "aasf"]
            for decomp in decompositions:
                n_proposals = 1 if decomp == "full" else 10
                pareto_solver = ParetoSolver(problem, pop=100, n_gen=100, seed=seed)
                optimize(solver, lambda: pareto_solver.get_adaptations(decomposition=decomp, seed=seed), n_proposals)
    return runtimes, results

def get_utilities_for_adaptations(adaptations, utility_functions):
    """
    Get the utilities of the given adaptations.
    
    Args:
        adaptations: A dict of the adaptations (key: adaptation_id, value: adaptation).
        utility_functions: A dict of the utility functions (key: utility_id, value: utility function).

    Returns:
        A dict of the utilities (key: utility_id, value: dict of the utilities
        for the adaptations (key: adaptation_id, value: utility)).
    """
    utilities = {}
    for utility_id, get_utility in utility_functions.items():
        utilities[utility_id] = {}
        for adaptation_id, adaptation in adaptations.items():
            utilities[utility_id][adaptation_id] = get_utility(adaptation)
    return utilities

def get_expected_utility(preference_criteria: list[OBJECTIVE], utility_functions: list[UTILITY_FUNCTION], n_trials: int = 1000, seed: int = 42) -> float:
    """
    Get the expected utility of the given preference criteria.
    
    Args:
        preference_criteria: A list of the preference criteria.

    Returns:
        The expected utility.
    """
    # Get problem
    problem = LayoutProblem(objectives=preference_criteria, weights=1/2)

    # Initialize random solver
    random_solver = RandomSolver(problem=problem, seed=seed)

    # Get random adaptations
    random_adaptation_layouts = random_solver.get_adaptations(n_adaptations=n_trials)

    # Get utilities
    utilities = get_utilities_for_adaptations(
        {
            adaptation_id: adaptation
            for adaptation_id, adaptation in enumerate(random_adaptation_layouts)
        },
        {
            utility_id: utility_function
            for utility_id, utility_function in enumerate(utility_functions)
        }
    )

    # Get expected utility
    expected_utility = np.mean([utility for adaptation_utilities in utilities.values() for utility in adaptation_utilities.values()])

    return expected_utility