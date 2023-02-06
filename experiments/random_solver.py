"""Functions to create a random solver."""

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

from .problem import LayoutProblem
import numpy as np
import networking.layout
from .solver import Solver

class RandomSolver(Solver):
    """A random solver for the layout problem."""

    def __init__(self, problem: LayoutProblem, seed: int = 42):
        """Initialize the solver."""
        super().__init__(problem)
        self.rng = np.random.default_rng(seed)

    def get_adaptations(self, n_adaptations: int = 1) -> list[networking.layout.Layout]:
        """Returns one or multiple random adaptations in the design space defined in the problem."""
        random_adaptations = self.problem.xl + self.rng.random((n_adaptations, self.problem.n_var)) * (
            self.problem.xu - self.problem.xl
        )
        if n_adaptations == 1:
            return [self.problem._x_to_layout(random_adaptations[0])]
        return [self.problem._x_to_layout(x) for x in random_adaptations]