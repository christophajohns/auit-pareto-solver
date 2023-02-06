"""Abstract solver class."""

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
import networking.layout

class Solver:
    """An abstract solver for the layout problem."""

    def __init__(self, problem: LayoutProblem):
        """Initialize the solver."""
        self.problem = problem

    def get_adaptations(self) -> list[networking.layout.Layout]:
        """Returns a list with one or multiple adaptations in the design space defined in the problem."""
        raise NotImplementedError