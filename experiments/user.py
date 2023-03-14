"""Functions to create computational user models implementing various utility functions."""

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

from .config import OBJECTIVE, UTILITY_FUNCTION
from .problem import get_objective_function
from typing import Optional, Union, Callable, List
import networking.layout
import numpy as np
import uuid


def get_utility_function(objectives: list[OBJECTIVE], weights: Optional[Union[list[float], float]] = None) -> UTILITY_FUNCTION:
    """Returns a utility function representing a weighted sum of the given
    objectives using the specified weights.

    Args:
        objectives: The objectives to include.
        (optional) weights: The weights for the objectives.
            If no weights are specified, equal weights are used.
    """
    if not isinstance(weights, np.ndarray) and not weights:
        weights = 1 / len(objectives)

    if isinstance(weights, float):
        weights = [weights] * len(objectives)

    objective_functions = [
        get_objective_function(objective=objective)
        for objective in objectives
    ]

    def get_utility(adaptation: networking.layout.Layout, verbose: bool = False) -> float:
        """Returns the utility score for a given adaptation.
        Maximum value is 1, minimum value is 0.
        
        Args:
            adaptation: The adaptation to evaluate.
        """
        if verbose:
            total_cost = 0.
            for (weight, objective, get_objective_score) in zip(weights, objectives, objective_functions):
                print(f"{objective}:", get_objective_score(adaptation))
                print(f"Weighted:", weight * get_objective_score(adaptation))
                total_cost += weight * get_objective_score(adaptation)
            print("Cost:", total_cost)
        return 1 - sum(
            [
                weight * get_objective_score(adaptation)
                for (weight, get_objective_score) in zip(weights, objective_functions)
            ]
        )

    return get_utility


def get_utility_functions(objectives: list[OBJECTIVE], weights: Optional[Union[list[float], float]] = None, n_functions: int = 100, seed: int = 42) -> List[UTILITY_FUNCTION]:
    """Returns a list of utility functions representing a weighted sum of the given
    objectives using the specified weights.

    Args:
        objectives: The objectives to include.
        (optional) weights: The weights for the objectives.
            If no weights are specified, random weights sampled from a uniform
            distribution ranging from 0 to 1 for each function invidivdually are used.
        (optional) n_functions: The number of utility functions to return.
        (optional) seed: The random seed to use.
    """
    # Create a random number generator
    rng = np.random.default_rng(seed=seed)

    # Generate random weights by sampling from a uniform distribution
    # ranging from 0 to 1 if none are specified
    population_weights = [
        rng.uniform(size=len(objectives))
        for _ in range(n_functions)
    ] if not weights else [weights] * n_functions

    # Generate utility functions using the weights
    utility_functions = [
        get_utility_function(objectives=objectives, weights=weights)
        for weights in population_weights
    ]

    # Return the utility functions
    return utility_functions


def get_utility_functions_for_different_seeds(preference_criteria, n_functions=100, seed=42):
    utility_functions = {}
    sample_utility_functions = get_utility_functions(
        objectives=preference_criteria,
        n_functions=n_functions,
        seed=seed
    )
    for utility_function in sample_utility_functions:
        utility_id = str(uuid.uuid4())
        utility_functions[utility_id] = utility_function

    return utility_functions