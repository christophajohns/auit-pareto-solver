"""Server for the Python solver backend."""

import zmq
import sys
from networking.messages import (
    HelloResponse,
    ErrorResponse,
    OptimizationResponse,
    EvaluationResponse,
    to_json,
    from_json,
)
import AUIT
from solver import optimize_layout


def handle_request(request_type, request_data):
    """Return a tuple of the response type and the response data."""
    # If request type is HelloRequest, return a HelloResponse
    if request_type == "H":
        print("Received a HelloRequest")
        return "h", HelloResponse()
    # If request type is OptimizationRequest, solve the optimization problem and
    # return an OptimizationResponse
    if request_type == "O":
        print("Received an OptimizationRequest")
        print("request_data:", request_data)
        solutions = optimize_layout(
            request_data.n_objectives,
            request_data.n_constraints,
            request_data.initial_layout,
        )
        return "o", OptimizationResponse(solutions=solutions)
    if request_type == "P":
        print("Received a Problem Layout")
        return "h", None
    # If request type is EvaluationRequest, evaluate the candidate layouts and
    # return an EvaluationResponse
    # NOTE: This should be implemented by the AUIT server and is not needed for the Python backend
    if request_type == "E":
        print("Received an EvaluationRequest")
        costs, violations = AUIT.evaluate_layouts(request_data.layouts)
        return "e", EvaluationResponse(costs=costs, violations=violations)
    # If request type is unknown, return an ErrorResponse
    else:
        print("Received an unknown request type: %s" % request_type)
        return "x", ErrorResponse(error="Unknown request type: %s" % request_type)


def run_server(port=5555):
    """Run the server."""
    # Create a context and a socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")

    # Loop forever
    while True:

        # Print listening message
        print(f"Listening on port {port}...")

        # Receive a request
        request = socket.recv_multipart()

        # Parse the request
        request_type = request[0].decode("utf-8")
        request_data = from_json(request_type, request[1].decode("utf-8"))

        # Handle the request
        response_type, response_data = handle_request(request_type, request_data)

        # Send the response
        if response_data is not None:
            print("Sending response...")
            socket.send_multipart(
                [response_type.encode("utf-8"), to_json(response_data).encode("utf-8")]
            )
        else:
            socket.send_string("ok")


def main():
    """Main function."""
    # Get port number from command line argument named "port" or "p"
    if len(sys.argv) == 1:  # if no arguments are given
        port = 5555
    else:
        if sys.argv[1] == "-p" or sys.argv[1] == "--port":
            port = int(sys.argv[2])
        else:
            raise ValueError("Unknown command line argument: %s" % sys.argv[1])

    # Run the server
    run_server(port)


if __name__ == "__main__":
    main()
