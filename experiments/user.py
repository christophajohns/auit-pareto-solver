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

from .config import OBJECTIVE
from .problem import get_objective_function
from typing import Optional, Union, Callable
import networking.layout


def get_utility_function(objectives: list[OBJECTIVE], weights: Optional[Union[list[float], float]]) -> Callable[[networking.layout.Layout, Optional[bool]], float]:
    """Returns a utility function representing a weighted sum of the given
    objectives using the specified weights.

    Args:
        objectives: The objectives to include.
        (optional) weights: The weights for the objectives.
            If no weights are specified, equal weights are used.
    """
    if not weights:
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
