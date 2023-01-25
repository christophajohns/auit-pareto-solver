"""Classes and functions for the multi-objective optimization of layouts."""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.visualization.scatter import Scatter
import client
import networking.layout
import networking.element

# Disable pymoo warnings
from pymoo.config import Config

Config.warnings["not_compiled"] = False


# Experiment Settings:
# Optimization Method: NSGA-III vs Weighted Sum --> NSGA-III (ref_dirs: Riesz, pop: 1000, n_gen: 100), WS on true PF with w_1 = w_2 = 0.5
# Objective Formulations: Discrete Hand Reachability Objective --> NSGA-III (ref_dirs: Riesz, pop: 1000, n_gen: 100), WS on true PF with w_1 = w_2 = 0.5
# Algorithms: NSGA-III vs U-NSGA-III vs SMSEMOA --> NSGA-III (ref_dirs: Riesz, pop: 4, n_gen: 100), U-NSGA-III (ref_dirs: Riesz, pop: 4, n_gen: 100), SMSEMOA (ref_dirs: Riesz, pop: 4, n_gen: 100)
# Decompositions: WS vs Tchebicheff vs AASF --> w_1 = 0.25, w_2 = 0.75; w_1 = w_2 = 0.5; w_1 = 0.75, w_2 = 0.25 with AASF(eps=1e-10, beta=25) to provide sharper direction towards 45 deg line


class LayoutProblem(Problem):
    """A multi-objective optimization problem for layouts."""

    def __init__(
        self,
        n_objectives: int,
        n_constraints: int,
        initial_layout: networking.layout.Layout,
        socket,
        **kwargs,
    ):
        """Initialize the problem."""
        # Calculate the number of variables
        n_variables = (
            initial_layout.n_elements * 7
        )  # 3 position variables + 4 rotation variables

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
            n_obj=n_objectives,
            n_ieq_constr=n_constraints,
            xl=xlower,
            xu=xupper,
            **kwargs,
        )

        # Store the socket
        self.socket = socket

    def _evaluate(self, x: np.ndarray, out, *args, **kwargs):
        """Evaluate the problem."""
        # Convert the decision variables to a list of layouts
        layouts = [self._x_to_layout(x[i]) for i in range(x.shape[0])]

        # Send the layouts to the server and receive the costs
        response_type, response_data = client.send_costs_request(self.socket, layouts)

        # Check if the response is an EvaluationResponse
        if response_type == "e":
            # Transform costs to a numpy array by storing
            # only the value of the costs for each layout's costs
            costs = np.array(
                [
                    [cost for cost in layout_costs]
                    for layout_costs in response_data.costs
                ]
            )

            # Set the objectives
            out["F"] = costs

            # If the problem is constrained, check the constraints
            if self.n_ieq_constr > 0:

                # Transform violations to a numpy array
                violations = np.array(
                    [
                        [violation for violation in layout_constraint_violations]
                        for layout_constraint_violations in response_data.violations
                    ]
                )

                # Set the constraint violations
                out["G"] = violations

        # If response type is unknown, print a message
        else:
            print("Received an unknown response type: %s" % response_type)

    def _x_to_layout(self, x):
        """Convert the decision variables to a layout."""
        # Create a list of elements
        elements = []
        for i in range(0, len(x), 7):
            elements.append(
                networking.element.Element(
                    position=networking.element.Position(x=x[i], y=x[i + 1], z=x[i + 2]),
                    rotation=networking.element.Rotation(x=x[i + 3], y=x[i + 4], z=x[i + 5], w=x[i + 6]),
                )
            )

        # Create and return the layout
        return networking.layout.Layout(elements=elements)


# Function to create an algorithm instance
def get_algorithm(n_objectives: int):
    """Create an algorithm instance."""
    # set population size
    pop_size = 20  # Exp. 1-2: 1000, Exp. 3: 4

    # create the reference directions to be used for the optimization
    # ref_dirs = get_reference_directions(
    #     "uniform", n_objectives, n_partitions=pop_size
    # )  # Exp. 3
    ref_dirs = get_reference_directions("energy", n_objectives, pop_size, seed=1)

    # create the algorithm object
    # algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)  # Exp. 1-3
    algorithm = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs)  # Exp. 3
    # algorithm = SMSEMOA(pop_size=pop_size, ref_dirs=ref_dirs)  # Exp. 3
    # algorithm = RVEA(pop_size=pop_size, ref_dirs=ref_dirs)  # Exp. 3

    return algorithm


# Function to generate the Pareto optimal layouts (i.e., the Pareto front)
def generate_pareto_optimal_layouts(
    n_objectives: int,
    n_constraints: int,
    initial_layout: networking.layout.Layout,
    socket,
    reduce=False,
    plot=False,
    save=False,
):
    """Generate the Pareto optimal layouts.

    Args:
        n_objectives: The number of objectives.
        initial_layout: The initial layout.
        socket: The socket to use for communication with the server.
        reduce: Whether to reduce the Pareto front using the high tradeoff points algorithm.
        plot: Whether to plot the Pareto front.
    """
    # Create the problem
    problem = LayoutProblem(
        n_objectives=n_objectives,
        n_constraints=n_constraints,
        initial_layout=initial_layout,
        socket=socket,
    )

    # Create the algorithm
    algorithm = get_algorithm(n_objectives)

    # Create the termination criterion
    termination = get_termination("n_gen", 100)  # Exp. 1-3: 100

    # Run the optimization
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        # verbose=True,
        save_history=True,
        copy_algorithm=False,
    )

    # Print the results
    print("Pareto front: %s" % res.F)
    print("Non-dominated solutions: %s" % res.X)

    # Save the results
    if save:
        algorithm_name = algorithm.__class__.__name__
        np.save(
            f"examples/neck_and_arm_angle/algorithm/{algorithm_name}_pareto_front.npy",
            res.F,
        )
        np.save(
            f"examples/neck_and_arm_angle/algorithm/{algorithm_name}_non_dominated_solutions.npy",
            res.X,
        )

    scatterplot = Scatter(title="Pareto front")
    scatterplot.add(res.F, alpha=0.5)

    # If a single global optimum is found, return it
    if res.F.shape[0] == 1:
        if (
            res.X.shape[0] == 1
        ):  # If the array is 1D (i.e., single-objective; e.g., res.X: [-2.1 -2.4 13.4  0.1  0.8  0.4  0.6])
            return [problem._x_to_layout(res.X[0])]
        # else if the array is 2D (i.e., multi-objective; e.g., res.X: [[-2.1 -2.4 13.4  0.1  0.8  0.4  0.6]])
        return [problem._x_to_layout(res.X)]

    # If the reduce flag is set to True...
    if reduce:
        # ...reduce the set of Pareto optimal layouts to the high tradeoff points
        htp = HighTradeoffPoints()
        points_of_interest = htp(res.F)  # This is a boolean array

        # Add the high tradeoff points to the scatterplot
        scatterplot.add(
            res.F[points_of_interest], s=40, facecolors="none", edgecolors="r"
        )

        # If the Pareto front should be plotted, plot it
        if plot:
            scatterplot.show()

        # Return the Pareto optimal layouts
        return [problem._x_to_layout(x) for x in res.X[points_of_interest]]

    # If the Pareto front should be plotted, plot it
    if plot:
        scatterplot.show()

    # Otherwise, return all the Pareto optimal layouts
    return [problem._x_to_layout(x) for x in res.X]
