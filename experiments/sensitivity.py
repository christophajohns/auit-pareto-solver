"""Functions to conduct sensitivity analysis on the objective and utility functions."""

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

import numpy as np
import pandas as pd
from typing import List
import AUIT
from experiments.problem import LayoutProblem
from experiments.pareto_solver import ParetoSolver

def get_object_locations(number_of_objects: int, interaction_volume_radius: float = 2, seed: int = 42) -> List[AUIT.networking.element.Position]:
    """Return the locations of a number of objects in a cube around the origin."""
    rng = np.random.default_rng(seed)
    # Get the locations of the objects
    object_locations_coordinates = rng.uniform(
        low=-interaction_volume_radius,
        high=interaction_volume_radius,
        size=(number_of_objects, 3),
    )
    object_locations = [
        AUIT.networking.element.Position(
            x=object_locations_coordinates[i, 0],
            y=object_locations_coordinates[i, 1],
            z=object_locations_coordinates[i, 2],
        )
        for i in range(number_of_objects)
    ]
    return object_locations

def get_association_scores(number_of_objects: int, association_score_variance: float, seed: int = 42) -> List[dict[str, float]]:
    """Return the positive and negative association scores for a number of objects.
    
    The association scores are normally distributed around 0.5 with the specified variance
    and clipped between 0 and 1.
    """
    rng = np.random.default_rng(seed)
    association_scores = [
        {
            "positive_association_score": min(max(rng.normal(loc=0.5, scale=association_score_variance), 0.), 1.),
            "negative_association_score": min(max(rng.normal(loc=0.5, scale=association_score_variance), 0.), 1.),
        }
        for i in range(number_of_objects)
    ]
    return association_scores

def get_clutter_association_grid(number_of_objects_params: np.ndarray, association_score_variance_params: np.ndarray, seed: int = 42) -> List[dict]:
    """Get the clutter association grid for a number of objects and association score variance parameters.
    
    The clutter association grid is a list of dictionaries with the following keys:
    - number_of_objects: The number of objects in the grid.
    - association_score_variance: The variance of the association scores.
    - objects: The locations and positive and negative assocation scores of the generated objects.
    """
    clutter_association_grid = []
    for number_of_objects in number_of_objects_params:
        for association_score_variance in association_score_variance_params:
            object_locations = get_object_locations(number_of_objects=number_of_objects, seed=seed)
            association_scores = get_association_scores(number_of_objects=number_of_objects, association_score_variance=association_score_variance, seed=seed)
            clutter_association_grid.append({
                "number_of_objects": int(number_of_objects),
                "association_score_variance": float(association_score_variance),
                "objects": [
                    {
                        "position": object_locations[i],
                        **association_scores[i],
                    }
                    for i in range(number_of_objects)
                ],
            })
    return clutter_association_grid

def get_minimum_semantic_costs_for_grid(association_dicts: List[dict], seed: int = 42) -> List[float]:
    """Get the minimum semantic costs for a clutter association grid."""
    minimum_semantic_costs = []
    for associations_dict in association_dicts:
        get_semantic_cost = lambda adaptation: sum([AUIT.get_semantic_cost(element=item, association_dict=associations_dict) for item in adaptation.items])
        # Determine the minimum semantic cost via an NSGA-III single-objective solver from pymoo
        layout_problem = LayoutProblem(objectives=["semantics"])
        layout_problem.objective_functions = [get_semantic_cost]
        solver = ParetoSolver(problem=layout_problem, seed=seed)
        optimal_adaptations = solver.get_adaptations(seed=seed)
        minimum_semantic_cost = get_semantic_cost(optimal_adaptations[0])
        minimum_semantic_costs.append(minimum_semantic_cost)
    return minimum_semantic_costs

def get_semantic_costs_grid_dataframe(association_dicts: List[dict], semantic_costs: List[float]) -> pd.DataFrame:
    """Get a DataFrame with the semantic costs for a clutter association grid."""
    semantic_costs_grid = [
        {
            "number_of_objects": associations_dict["number_of_objects"],
            "association_score_variance": associations_dict["association_score_variance"],
            "minimum_semantic_cost": semantic_cost,
        }
        for associations_dict, semantic_cost in zip(association_dicts, semantic_costs)
    ]
    semantic_costs_grid_df = pd.DataFrame(semantic_costs_grid)
    return semantic_costs_grid_df