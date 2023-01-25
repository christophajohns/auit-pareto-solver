"""Client for the AUIT server."""

import zmq
import sys
from networking.messages import (
    HelloRequest,
    OptimizationRequest,
    EvaluationRequest,
    to_json,
    from_json,
)
from networking.layout import Layout
import AUIT


def handle_response(response_type, response_data):
    """Handle a response."""
    # If response type is HelloResponse, print a message
    if response_type == "h":
        print("Received a HelloResponse")
    # If response type is OptimizationResponse, print a message
    # NOTE: This should be implemented by the AUIT client.
    elif response_type == "o":
        print("Received an OptimizationResponse")
        AUIT.handle_optimization_response(response_data)
    # If response type is EvaluationResponse, print a message
    elif response_type == "e":
        print("Received an EvaluationResponse")
        # print("Costs:", response_data.costs)
        # print("Violations:", response_data.violations)
    # If response type is ErrorResponse, print the error
    elif response_type == "x":
        print("Received an ErrorResponse: %s" % response_data.error)
    # If response type is unknown, print a message
    else:
        print("Received an unknown response type: %s" % response_type)


def send_request(socket, request_type, request_data):
    """Send a request and return the response."""
    # Send the request
    socket.send_multipart(
        [request_type.encode("utf-8"), to_json(request_data).encode("utf-8")]
    )

    # Receive a response
    response = socket.recv_multipart()

    # Parse the response
    response_type = response[0].decode("utf-8")
    response_data = from_json(response_type, response[1].decode("utf-8"))

    # Handle the response
    handle_response(response_type, response_data)

    # Return the response
    return response_type, response_data


def send_hello_request(socket):
    """Send a HelloRequest and return the response."""
    # Construct the request
    request_type = "H"
    request_data = HelloRequest()

    # Send the request and return the response
    return send_request(socket, request_type, request_data)


# NOTE: This should be implemented by the AUIT client.
def send_optimization_request(
    socket, n_objectives: int, n_constraints: int, initial_layout: Layout
):
    """Send an OptimizationRequest and return the response."""
    # Construct the request
    request_type = "O"
    request_data = OptimizationRequest(
        n_objectives=n_objectives,
        n_constraints=n_constraints,
        initial_layout=initial_layout,
    )

    # Send the request and return the response
    return send_request(socket, request_type, request_data)


def send_costs_request(socket, layouts):
    """Send an EvaluationRequest and return the response."""
    # Print a message
    print("Sending EvaluationRequest...")

    # Construct the request
    request_type = "E"
    request_data = EvaluationRequest(
        layouts=layouts,
    )

    # Send the request and return the response
    return send_request(socket, request_type, request_data)


def main():
    """Main function."""
    # Get port number from command line argument named "port" or "p"
    if len(sys.argv) == 1:  # if no arguments are given
        port = 5556
    else:
        if sys.argv[1] == "-p" or sys.argv[1] == "--port":
            port = int(sys.argv[2])
        else:
            raise ValueError("Unknown command line argument: %s" % sys.argv[1])

    # Create a context and a socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")

    # Send a HelloRequest
    # print("Sending HelloRequest...")
    # response_type, response_data = send_hello_request(socket)

    # Handle the response
    # handle_response(response_type, response_data)

    # Send an EvaluationRequest
    print("Sending EvaluationRequest...")
    response_data = send_costs_request(socket, None)
    print(response_data)

    # Handle the response
    # handle_response(response_type, response_data)


if __name__ == "__main__":
    main()