"""Classes for messages sent between the server and the client.

The message types for which both request and response defined here are:
    - Hello
    - Optimization
    - Evaluation
    - Error

The JSON structures for the messages are defined in the from_json and to_json
functions below.
"""

from dataclasses import dataclass
import json
from typing import List
from .layout import Layout


@dataclass
class Message:
    """A message is a JSON object sent between the server and the client."""

    pass


@dataclass
class Request(Message):
    """A request is a message sent from the client to the server."""

    pass


@dataclass
class Response(Message):
    """A response is a message sent from the server to the client."""

    pass


@dataclass
class HelloRequest(Request):
    """A hello request is a request sent from the client to the server
    to establish a connection."""

    pass


@dataclass
class HelloResponse(Response):
    """A hello response is a response sent from the server to the client
    to establish a connection."""

    pass


@dataclass
class OptimizationRequest(Request):
    """An optimization request is a request sent from AUIT to the solver
    to receive the Pareto optimal solutions to a layout optimization problem."""

    n_objectives: int  # Number of objectives
    n_constraints: int  # Number of inequality constraints (<= 0)
    initial_layout: Layout  # Initial layout


@dataclass
class OptimizationResponse(Response):
    """An optimization response is a response sent from the solver to AUIT
    that contains the Pareto optimal solutions to the layout optimization problem."""

    solutions: List[Layout]


@dataclass
class EvaluationRequest(Request):
    """An evaluation request is a request sent from the solver to AUIT
    to get the vectors of costs for a list of layouts including the
    costs associated with constraint violations."""

    layouts: List[Layout]


@dataclass
class EvaluationResponse(Response):
    """An evaluation response is a response sent from AUIT to the solver
    containing the cost vectors for the list of layouts contained in the request,
    including the costs associated with constraint violations."""

    costs: List[List[float]]
    violations: List[List[float]]


@dataclass
class ErrorResponse(Response):
    """An error response is a response sent from the server to the client
    to report an error."""

    error: str


def from_json(message_type: str, message_data: str) -> Message:
    """Return a message from a JSON string based on the provided type.

    Args:
        request_type: The type of the message.
        json_str: The JSON string representing the request data.

    The JSON strings for the messages defined here are:
        - HelloRequest ("H"): {}
        - HelloResponse ("h"): {}
        - OptimizationRequest ("O"): {
            "nObjectives": <int>,
            "nConstraints": <int>,
            "initialLayout": {
                "elements": [
                    {
                        "id": <str>,
                        "position": [<float>, <float>, <float>],
                        "rotation": [<float>, <float>, <float>, <float>],
                    },
                    ...
                ],
            },
        }
        - OptimizationResponse ("o"): {
            "solutions": [
                {
                    "elements": [
                        {
                            "id": <str>,
                            "position": [<float>, <float>, <float>],
                            "rotation": [<float>, <float>, <float>, <float>],
                        },
                        ...
                    ],
                },
                ...
            ],
        }
        - EvaluationRequest ("E"): {
            "layouts": [
                {
                    "elements": [
                        {
                            "id": <str>,
                            "position": [<float>, <float>, <float>],
                            "rotation": [<float>, <float>, <float>, <float>],
                        },
                        ...
                    ],
                },
                ...
            ],
        }
        - EvaluationResponse ("e"): {
            "costs": [[<float>, ...], ...],
            "violations": [[<float>, ...], ...],
        }
        - ErrorResponse ("x"): {
            "error": <str>,
        }

    """
    data = json.loads(message_data)
    if message_type == "P":
        print(data)
        return data
    if message_type == "H":
        return HelloRequest()
    elif message_type == "h":
        return HelloResponse()
    elif message_type == "O":
        return OptimizationRequest(
            n_objectives=data["nObjectives"],
            n_constraints=data["nConstraints"] if "nConstraints" in data else 0,
            initial_layout=Layout.from_dict(data["initialLayout"]),
        )
    elif message_type == "o":
        return OptimizationResponse(
            solutions=[Layout.from_dict(layout) for layout in data["solutions"]]
        )
    elif message_type == "E":
        return EvaluationRequest(
            layouts=[Layout.from_dict(layout) for layout in data["layouts"]]
        )
    elif message_type == "e":
        return EvaluationResponse(
            costs=[[cost for cost in layout_costs] for layout_costs in data["costs"]],
            violations=[
                [cost for cost in layout_constraint_violation_costs]
                for layout_constraint_violation_costs in data["violations"]
            ],
        )
    elif message_type == "x":
        return ErrorResponse(error=data["error"])
    else:
        raise ValueError("Unknown message type: %s" % message_type)


def to_json(message: Message) -> str:
    """Return a JSON string from a message.

    The JSON strings for the messages defined here are:
        - HelloRequest: {}
        - HelloResponse: {}
        - OptimizationRequest: {
            "nObjectives": <int>,
            "nConstraints": <int>,
            "initialLayout": {
                "elements": [
                    {
                        "id": <str>,
                        "position": [<float>, <float>, <float>],
                        "rotation": [<float>, <float>, <float>, <float>],
                    },
                    ...
                ],
            },
        }
        - OptimizationResponse: {
            "solutions": [
                {
                    "elements": [
                        {
                            "id": <str>,
                            "position": [<float>, <float>, <float>],
                            "rotation": [<float>, <float>, <float>, <float>],
                        },
                        ...
                    ],
                },
                ...
            ],
        }
        - EvaluationRequest: {
            "layouts": [
                {
                    "elements": [
                        {
                            "id": <str>,
                            "position": [<float>, <float>, <float>],
                            "rotation": [<float>, <float>, <float>, <float>],
                        },
                        ...
                    ],
                },
                ...
            ],
        }
        - EvaluationResponse: {
            "costs": [[<float>, ...], ...],
            "violations": [[<float>, ...], ...],
        }
        - ErrorResponse: {
            "error": <str>,
        }

    """
    if isinstance(message, HelloRequest):
        return json.dumps({})
    elif isinstance(message, HelloResponse):
        return json.dumps({})
    elif isinstance(message, OptimizationRequest):
        return json.dumps(
            {
                "nObjectives": message.n_objectives,
                "nConstraints": message.n_constraints,
                "initialLayout": message.initial_layout.__dict__(),
            }
        )
    elif isinstance(message, OptimizationResponse):
        return json.dumps(
            {
                "solutions": [layout.__dict__() for layout in message.solutions],
            }
        )
    elif isinstance(message, EvaluationRequest):
        return json.dumps(
            {
                "layouts": [layout.__dict__() for layout in message.layouts],
            }
        )
    elif isinstance(message, EvaluationResponse):
        return json.dumps(
            {
                "costs": [
                    [cost for cost in layout_costs] for layout_costs in message.costs
                ],
                "violations": [
                    [cost for cost in layout_constraint_violation_costs]
                    for layout_constraint_violation_costs in message.violations
                ],
            }
        )
    elif isinstance(message, ErrorResponse):
        return json.dumps({"error": message.error})
    else:
        raise ValueError("Unknown message type: %s" % message)
