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
from typing import Union


def get_random_adaptations(problem: LayoutProblem, n_adaptations: int = 1) -> Union[networking.layout.Layout, list[networking.layout.Layout]]:
    """Returns one or multiple random adaptations in the design space defined in the problem."""
    random_adaptations = problem.xl + np.random.rand(n_adaptations, problem.n_var) * (
        problem.xu - problem.xl
    )
    if n_adaptations == 1:
        return problem._x_to_layout(random_adaptations[0])
    return [problem._x_to_layout(x) for x in random_adaptations]