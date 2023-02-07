"""Functions to create a weighted sum solver."""

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
from typing import Union, Optional, Literal
from .pareto_solver import ParetoSolver
from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.core.result import Result as PymooResult
from pymoo.optimize import minimize

class WeightedSumSolver(ParetoSolver):
    """A weighted sum solver for the layout problem."""

    def __init__(self, problem: LayoutProblem, weights: Optional[Union[list[float], float]], algo: Literal["unsga3", "nm"] = "unsga3", **kwargs):
        """Initialize the solver."""
        super().__init__(problem, **kwargs)
        if not weights:
            self.weights = 1 / problem.n_obj

        if isinstance(weights, float):
            self.weights = [weights] * problem.n_obj
        
        self.weights = weights
        self.algo = algo

    def _get_nelder_mead_solutions(self) -> PymooResult:
        """Returns the solutions of the Nelder Mead algorithm."""
        soo_problem = self.problem.get_soo_problem(weights=self.weights)
        algorithm = NelderMead(seed=self.seed, pop_size=self.pop)
        res = minimize(
            soo_problem,
            algorithm,
            termination=("n_gen", self.n_gen),
            seed=self.seed,
        )
        return res

    def get_adaptations(self, verbose: bool = False) -> list[networking.layout.Layout]:
        """Returns a single adaptation in the design space defined in the problem that is defined by
        an equally weighted sum over the objectives."""
        if self.algo == "unsga3":
            res = super()._get_solutions(verbose=verbose)
            ws = WeightedSum()
            weighted_sum_layout = self.problem._x_to_layout(res.X[ws.do(res.F, weights=self.weights).argmin()])
            return [weighted_sum_layout]
        if self.algo == "nm":
            res = self._get_nelder_mead_solutions()
            adaptation = self.problem._x_to_layout(res.X[res.F.argmin()]) if res.F.size > 1 else self.problem._x_to_layout(res.X)
            return [adaptation]
        raise ValueError("Unknown algorithm.")

        