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

import numpy as np
from pymoo.core.problem import Problem
import networking.layout
import networking.element
from typing import Callable
from .config import OBJECTIVE, OBJECTIVE_FUNCTIONS



def get_objective_function(objective: OBJECTIVE) -> Callable[[networking.layout.Layout], float]:
    """Returns the specified objective function.
    
    Args:
        objective: The objective function to retrieve.
    """
    if objective not in OBJECTIVE_FUNCTIONS:
        raise Exception("objective is not found or implemented")
    return OBJECTIVE_FUNCTIONS.get(objective)

class LayoutProblem(Problem):
    """A multi-objective optimization problem for layouts."""

    def __init__(
        self,
        objectives: list[OBJECTIVE],
        **kwargs,
    ):
        """Initialize the problem."""
        # Calculate the number of variables
        n_variables = 7  # 3 position variables + 4 rotation variables

        # TODO: Set the lower and upper bounds:
        # Each position is bounded between -20 and 20 for x, y and z (This is arbitrary)
        # Each rotation is bounded between 0 and 1 for x, y, z and w
        xlower = [-20] * n_variables
        xupper = [20] * n_variables
        for i in range(3, n_variables, 7):
            for j in range(4):  # x, y, z and w
                xlower[i + j] = 0
                xupper[i + j] = 1

        # Call the superclass constructor
        super().__init__(
            n_var=n_variables,
            n_obj=len(objectives),
            xl=xlower,
            xu=xupper,
            **kwargs,
        )

        # Store the objective functions
        self.objective_functions = [
            get_objective_function(objective=objective)
            for objective in objectives
        ]

    def _evaluate(self, x: np.ndarray, out, *args, **kwargs):
        """Evaluate the problem."""
        # Convert the decision variables to a list of layouts
        layouts = [self._x_to_layout(x[i]) for i in range(x.shape[0])]

        # Evaluate the layouts
        # Transform costs to a numpy array by storing
        # only the value of the costs for each layout's costs
        costs = np.array(
            [
                get_cost(layout) for get_cost in self.objective_functions
                for layout in layouts
            ]
        )

        # Set the objectives
        out["F"] = costs

    def _x_to_layout(self, x):
        """Convert the decision variables to a layout."""
        # Create a list of items
        items = []
        for i in range(0, len(x), 7):
            items.append(
                networking.element.Element(
                    position=networking.element.Position(x=x[i], y=x[i + 1], z=x[i + 2]),
                    rotation=networking.element.Rotation(x=x[i + 3], y=x[i + 4], z=x[i + 5], w=x[i + 6]),
                )
            )

        # Create and return the layout
        return networking.layout.Layout(items=items)

    def _layout_to_x(self, layout: networking.layout.Layout):
        """Convert a layout to decision variables."""
        # Create a list of decision variables
        x = []
        for item in layout.items:
            x.append(item.position.x)
            x.append(item.position.y)
            x.append(item.position.z)
            x.append(item.rotation.x)
            x.append(item.rotation.y)
            x.append(item.rotation.z)
            x.append(item.rotation.w)

        # Return the decision variables
        return np.array(x)
    
    def is_valid(self, x):
        """Check if the decision variables are valid."""
        # Check if the decision variables are within the bounds
        return np.all(x >= self.xl) and np.all(x <= self.xu)

    def layout_is_valid(self, layout: networking.layout.Layout):
        """Check if the layout is valid."""
        # Check if the layout is in the design space
        return self.is_valid(self._layout_to_x(layout))
