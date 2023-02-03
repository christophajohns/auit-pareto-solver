"""Functions to create computational user models implementing various utility functions."""

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

import AUIT
import networking.element

from typing import Optional, Union, Callable, Literal

Adaptation = networking.element.Element
OBJECTIVE = Literal["neck", "shoulder", "torso", "reach"]

# Hyperparameters
EYE_POSITION = AUIT.networking.element.Position(x=0.0, y=0.0, z=0.0)
SHOULDER_JOINT_POSITION = AUIT.networking.element.Position(x=0.0, y=-0.3, z=0.0)
waist_position = AUIT.networking.element.Position(x=0.0, y=-0.7, z=0.0)

OBJECTIVE_FUNCTIONS = {
    "neck": lambda adaptation: AUIT.get_neck_ergonomics_cost(eye_position=EYE_POSITION, element=adaptation),
    "shoulder": lambda adaptation: AUIT.get_arm_ergonomics_cost(shoulder_joint_position=SHOULDER_JOINT_POSITION, element=adaptation),
    "torso": lambda adaptation: AUIT.get_torso_ergonomics_cost(waist_position=waist_position, element=adaptation),
    "reach": lambda adaptation: AUIT.get_arm_ergonomics_cost(shoulder_joint_position=SHOULDER_JOINT_POSITION, element=adaptation),
}

def get_objective_function(objective: OBJECTIVE) -> Callable[[Adaptation], float]:
    """Returns the specified objective function.
    
    Args:
        objective: The objective function to retrieve.
    """
    if objective not in OBJECTIVE_FUNCTIONS:
        raise Exception("objective is not found or implemented")
    return OBJECTIVE_FUNCTIONS.get(objective)

def get_utility_function(objectives: list[OBJECTIVE], weights: Optional[Union[list[float], float]]) -> Callable[[Adaptation, Optional[bool]], float]:
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

    def get_utility(adaptation: Adaptation, verbose: bool = False) -> float:
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
