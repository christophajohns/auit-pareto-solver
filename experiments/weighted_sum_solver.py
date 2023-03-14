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
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga3 import NSGA3
from optimization import PrintProgress
import numpy as np

class WeightedSumSolver(ParetoSolver):
    """A weighted sum solver for the layout problem."""

    def __init__(self, problem: LayoutProblem, weights: Optional[Union[list[float], float]] = None, algo: Literal["unsga3", "nm", "nsga3"] = "unsga3", **kwargs):
        """Initialize the solver."""
        super().__init__(problem, **kwargs)
        if isinstance(weights, list) and len(weights) == problem.n_obj:
            self.weights = weights

        elif not weights:
            self.weights = 1 / problem.n_obj

        if isinstance(weights, float):
            self.weights = [weights] * problem.n_obj

        self.algo = algo
        self.rng = np.random.default_rng(self.seed)

    def _get_nelder_mead_solutions(self, seed, verbose: bool = False) -> PymooResult:
        """Returns the solutions of the Nelder Mead algorithm."""
        soo_problem = self.problem.get_soo_problem(weights=self.weights)
        algorithm = NelderMead(seed=seed, pop_size=self.pop)
        res = minimize(
            soo_problem,
            algorithm,
            termination=("n_gen", self.n_gen),
            seed=seed,
            callback=PrintProgress(n_gen=self.n_gen) if verbose else lambda _: None,
        )
        return res
    
    def _get_soo_nsga3_solutions(self, seed, verbose: bool = False) -> PymooResult:
        """Returns the solutions of the single-objective problem using the NSGA-III algorithm."""
        soo_problem = self.problem.get_soo_problem(weights=self.weights)
        ref_dirs = get_reference_directions("energy", 1, self.pop, seed=seed)
        algorithm = NSGA3(pop_size=self.pop, ref_dirs=ref_dirs, seed=seed)
        termination = get_termination("n_gen", self.n_gen)
        res = minimize(
            soo_problem,
            algorithm,
            termination,
            seed=seed,
            callback=PrintProgress(n_gen=self.n_gen) if verbose else lambda _: None,
        )

        return res

    def get_adaptations(self, max_n_proposals: int = 1, verbose: bool = False) -> list[networking.layout.Layout]:
        """Returns a single adaptation in the design space defined in the problem that is defined by
        an equally weighted sum over the objectives."""
        if self.algo == "unsga3":
            res = super()._get_solutions(verbose=verbose)
            ws = WeightedSum()
            weighted_sum_layout = self.problem._x_to_layout(res.X[ws.do(res.F, weights=self.weights).argmin()])
            return [weighted_sum_layout]
        if self.algo in ["nm", "nsga3"]:
            # Generate max_n_proposals seeds
            seeds = self.rng.integers(0, 2**8, size=max_n_proposals)
            # Run the algorithm for each seed
            adaptations = []
            for seed in seeds:
                res = self._get_nelder_mead_solutions(seed=seed, verbose=verbose) if self.algo == "nm" else self._get_soo_nsga3_solutions(seed=seed, verbose=verbose)
                adaptation = self.problem._x_to_layout(res.X[res.F.argmin()]) if res.F.size > 1 else self.problem._x_to_layout(res.X)
                adaptations.append(adaptation)
            return adaptations
        raise ValueError("Unknown algorithm.")

        