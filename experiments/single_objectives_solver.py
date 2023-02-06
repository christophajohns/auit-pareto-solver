"""Functions to create a multiple single objectives solver."""

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
import networking.layout
from .pareto_solver import ParetoSolver
from pymoo.decomposition.weighted_sum import WeightedSum

class SingleObjectivesSolver(ParetoSolver):
    """A multiple single objectives solver for the layout problem."""

    def get_adaptations(self, verbose: bool = False) -> list[networking.layout.Layout]:
        """Returns a single adaptation in the design space defined in the problem that is defined by
        an equally weighted sum over the objectives."""
        res = super()._get_solutions(verbose=verbose)
        ws = WeightedSum()
        single_obj_weights = np.zeros((self.problem.n_obj, self.problem.n_obj))
        for i in range(self.problem.n_obj):
            single_obj_weights[i, i] = 1
        return [
            self.problem._x_to_layout(res.X[ws.do(res.F, weights=weights).argmin()])
            for weights in single_obj_weights
        ]

        