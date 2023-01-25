"""Solver for the PythonBackend.

It spins up a server that listens for optimization requests from AUIT
and responds with the Pareto optimal solutions to the layout optimization
problem."""

import server
import zmq
import optimization
import networking.layout


def optimize_layout(
    n_objectives: int, n_constraints: int, initial_layout: networking.layout.Layout
):
    """Return the Pareto optimal solutions to the layout optimization problem."""
    # Create a context and a socket
    AUIT_PORT = 5556
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{AUIT_PORT}")

    # Generate the Pareto optimal layouts
    layouts = optimization.generate_pareto_optimal_layouts(
        n_objectives, n_constraints, initial_layout, socket
    )

    # Return the Pareto optimal solutions
    return layouts


def main():
    """Main function."""
    return server.run_server(port=5555)


if __name__ == "__main__":
    main()
