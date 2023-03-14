"""Functions to create a Pareto solver."""

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
from typing import Union, Literal, Optional
from .solver import Solver
from optimization import PrintProgress, get_algorithm
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.decomposition.aasf import AASF
from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.core.result import Result as PymooResult
from pymoo.util.ref_dirs import get_reference_directions


class ParetoSolver(Solver):
    """A multi-objective solver for the layout problem."""

    def __init__(self, problem: LayoutProblem, pop: int = 100, n_gen: int = 100, seed: int = 42):
        """Initialize the solver."""
        super().__init__(problem)
        self.seed = seed
        self.pop = pop
        self.n_gen = n_gen

    def _get_solutions(self, verbose: bool = False) -> PymooResult:
        """Returns the Pareto front of the problem."""
        # Create the algorithm
        algorithm = get_algorithm(self.problem.n_obj, pop_size=self.pop, seed=self.seed)

        # Create the termination criterion
        termination = get_termination("n_gen", self.n_gen)

        # Run the optimization
        res = minimize(
            self.problem,
            algorithm,
            termination,
            seed=self.seed,
            callback=PrintProgress(n_gen=self.n_gen) if verbose else lambda _: None,
            # verbose=True,
            save_history=True,
            copy_algorithm=False,
        )

        return res

    def get_adaptations(self, decomposition: Optional[Literal['full', 'aasf', 'ws']] = None, verbose: bool = False, seed: int = 42) -> list[networking.layout.Layout]:
        """Returns one or multiple Pareto optimal adaptations in the design space defined in the problem."""
        # Get the Pareto front
        res = self._get_solutions(verbose=verbose)

        # If a single global optimum is found, return it
        if res.F.shape[0] == 1:
            if (
                res.X.shape[0] == 1
            ):  # If the array is 1D (i.e., single-objective; e.g., res.X: [-2.1 -2.4 13.4  0.1  0.8  0.4  0.6])
                return [self.problem._x_to_layout(res.X[0])]
            # else if the array is 2D (i.e., multi-objective; e.g., res.X: [[-2.1 -2.4 13.4  0.1  0.8  0.4  0.6]])
            return [self.problem._x_to_layout(res.X)]

        # If no decomposition is requested, return all the Pareto optimal layouts
        if not decomposition:
            return [self.problem._x_to_layout(x) for x in res.X]

        equal_weights = np.ones(self.problem.n_obj) / self.problem.n_obj

        # If weighted sum decomposition is requested, return the Pareto optimal layout decomposed via an equally weighted sum
        if decomposition == 'ws':
            decomp = WeightedSum()
            equal_weight_layout_ws = self.problem._x_to_layout(res.X[decomp.do(res.F, weights=equal_weights).argmin()])
            return [equal_weight_layout_ws]
        
        # Otherwise, set up AASF decomposition
        decomp = AASF(eps=0.0, beta=25)

        # If full decomposition is requested, return the Pareto optimal layout decomposed via equally weighted AASF
        if decomposition == 'full':
            equal_weight_layout_aasf = self.problem._x_to_layout(res.X[decomp.do(res.F, weights=equal_weights).argmin()])
            return [equal_weight_layout_aasf]

        # If AASF decomposition is requested, return Pareto optimal layouts decomposed via AASF
        # Return the layouts with well-spaced reference directions derived via
        # Riesz s-Energy
        N_REF_DIRS = 10
        ref_dirs = get_reference_directions(
            "energy", self.problem.n_obj, N_REF_DIRS, seed=seed
        )
        return [
            self.problem._x_to_layout(res.X[decomp.do(res.F, weights=ref_dir).argmin()])
            for ref_dir in ref_dirs
        ]   