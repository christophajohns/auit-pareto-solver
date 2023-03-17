"""Simulated AUIT frontend for testing the PythonBackend.

It spins up a server that listens for layout evaluation requests from the solver
and responds with the costs of the candidate layouts. It also spins up a client
that sends optimization requests to the solver and receives the Pareto optimal
solutions to the layout optimization problem.

The AUIT server and client need to adhere to the following protocol:

1. The server needs to listen for EvaluationRequests ("E") and send EvaluationResponses ("e")
    which indicate the cost vectors (objectives and constraints) for a given list of candidate layouts.
2. The client needs to send OptimizationRequests ("O") and listen for OptimizationResponses ("o")
    which indicate the Pareto optimal solutions to the layout optimization problem.

The EvaluationRequest and OptimizationRequest as well as the EvaluationResponse and
OptimizationResponse messages are defined in the networking.messages module.
"""

# Current issues:
# - The Pareto optimal solutions can be in front of or behind the user (no information about gaze direction)
# - The optimization request does not contain the user's eye position, arm length, and shoulder joint position (required for the cost functions)
# - The "at-arms-length" cost should instead be a constraint that needs to be communicated to the solver

import threading
import client
import server
import zmq
import math
import numpy as np
import networking.layout
import networking.element
from typing import List, Union


def handle_optimization_response(response_data):
    # Placeholder for displaying the solutions in 3D
    # or other handling of the response
    print("Solutions:")
    for solution in response_data.solutions:
        print(solution)


def get_pareto_optimal_solutions(
    n_objectives: int, n_constraints: int, layout: networking.layout.Layout, verbose=True
):
    """Return the Pareto optimal solutions to the layout optimization problem."""
    # Create a context and a socket
    SOLVER_PORT = 5555
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{SOLVER_PORT}")

    # Send a HelloRequest
    if verbose: print("Sending HelloRequest...")
    response_type, response_data = client.send_hello_request(socket)

    # Handle the response
    client.handle_response(response_type, response_data)

    # Send an OptimizationRequest
    if verbose: print("Sending OptimizationRequest...")
    response_type, response_data = client.send_optimization_request(
        socket, n_objectives, n_constraints, layout
    )

    # Handle the response
    client.handle_response(response_type, response_data)

    # Return the Pareto optimal solutions
    return response_data.solutions


def get_at_arms_length_cost(
    shoulder_joint_position: networking.element.Position,
    arm_length: float,
    element: networking.element.Element,
    tolerance: float = 0.01,
    decay: float = 0.5,
):
    """Return the at-arms-length cost of an element for a given shoulder joint position
    and arm length. The distance between the element and the shoulder joint position
    should be equal to the arm length with a given tolerance in percent of the arm length."""
    # Return cost of 1 if the arm length is zero to avoid division by zero
    if arm_length == 0:
        return 1.0

    # Calculate the distance between the shoulder joint and the element
    distance = math.sqrt(
        (shoulder_joint_position.x - element.position.x) ** 2
        + (shoulder_joint_position.y - element.position.y) ** 2
        + (shoulder_joint_position.z - element.position.z) ** 2
    )

    # Calculate difference between the distance and the arm length
    distance_difference = abs(distance - arm_length)

    # Calculate the at-arms-length cost
    at_arms_length_cost = 0.0 if distance <= arm_length + tolerance * arm_length else 1 - np.exp(-decay * (distance_difference - tolerance * arm_length))

    return at_arms_length_cost


def get_angle_between_vectors(vector1: List[float], vector2: List[float]):
    """Return the angle between two vectors as a multiple of pi attaching
    the vectors tail to tail."""
    # Calculate the angle between the two vectors
    # The angle is the arccos of the dot product of the two normalized vectors

    # If one of the vectors is a zero-vector, return 0
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0

    normalized_vector1 = vector1 / np.linalg.norm(vector1)
    normalized_vector2 = vector2 / np.linalg.norm(vector2)

    angle = math.acos(np.dot(normalized_vector1, normalized_vector2))

    # If the angle is undefined, return None
    if math.isnan(angle):
        return None

    return angle


def get_neck_ergonomics_cost(
    eye_position: networking.element.Position, element: networking.element.Element
):
    """
    Return the neck ergonomics cost of an element for a given eye position.

    The neck ergonomics cost is calculated as the angle between the vector
    from the eye position to the element and the vector from the eye position
    to the target's projection on the xz-plane at the eye position's height.
    """
    # If the element is directly under the eye position, return 1
    if (
        element.position.x == eye_position.x
        and element.position.z == eye_position.z
    ):
        return 1.0

    # Calculate the vector from the eye position to the element
    eye_to_element_vector = [
        element.position.x - eye_position.x,
        element.position.y - eye_position.y,
        element.position.z - eye_position.z,
    ]

    # Calculate the vector from the eye position to the target's projection
    # on the xz-plane at the eye position's height
    eye_to_target_projection_vector = [
        element.position.x - eye_position.x,
        0.0,
        element.position.z - eye_position.z,
    ]

    # Calculate the angle between the two vectors
    # The angle is the arccos of the ratio of the dot product of the two vectors
    # and the product of the lengths of the two vectors
    neck_angle = get_angle_between_vectors(
        eye_to_element_vector, eye_to_target_projection_vector
    )

    # If the angle is undefined, return 1
    if neck_angle == None:
        # print("Neck angle is undefined")
        return 1.0

    # Calculate the neck ergonomics cost normalized to [0, 1]
    neck_ergonomics_cost = neck_angle / math.pi / 2

    # Print neck angle in degrees
    # print(f"Neck angle: {neck_angle * 180 / math.pi}°")

    return neck_ergonomics_cost

def get_torso_ergonomics_cost(
    waist_position: networking.element.Position, element: networking.element.Element, verbose: bool = False
):
    """
    Return the torso ergonomics cost of an element for a given waist position.

    The torso ergonomics cost is calculated as the angle between the vector
    from the waist position to the element and the vector from the waist position
    to the target's projection on the xz-plane at the waist position's height.
    If the target is at or above the waist, the torso ergonomics cost is 0.
    """
    # Print parameters
    if verbose:
        print("Waist at:", waist_position)
        print("Element at:", element.position)

    # If the element is at or directly under the waist position, return 1
    if (
        element.position.x == waist_position.x and
        element.position.y <= waist_position.y and
        element.position.z == waist_position.z
    ):
        return 1.0

    # If the element is at or above waist level, return 0
    if element.position.y >= waist_position.y: return 0.0

    # Calculate the vector from the waist position to the element
    waist_to_element_vector = [
        element.position.x - waist_position.x,
        element.position.y - waist_position.y,
        element.position.z - waist_position.z,
    ]

    # Calculate the vector from the waist position to the target's projection
    # on the xz-plane at the waist position's height
    waist_to_target_projection_vector = [
        element.position.x - waist_position.x,
        0.0,
        element.position.z - waist_position.z,
    ]

    # Calculate the angle between the two vectors
    # The angle is the arccos of the ratio of the dot product of the two vectors
    # and the product of the lengths of the two vectors
    waist_angle = get_angle_between_vectors(
        waist_to_element_vector, waist_to_target_projection_vector
    )

    # If the angle is undefined, return 1
    if waist_angle == None:
        if verbose: print("Waist angle is undefined")
        return 1.0

    # Calculate the torso ergonomics cost normalized to [0, 1]
    torso_ergonomics_cost = waist_angle / math.pi / 2

    # Print waist angle in degrees
    if verbose: print(f"Waist angle: {waist_angle * 180 / math.pi}°")

    return torso_ergonomics_cost

def get_arm_angle(
    shoulder_joint_position: networking.element.Position,
    element: networking.element.Element,  
):
    """
    Return the angle of the arm trying to reach the element.
    """
    # Calculate the vector from the shoulder joint position to the element
    shoulder_to_element_vector = [
        element.position.x - shoulder_joint_position.x,
        element.position.y - shoulder_joint_position.y,
        element.position.z - shoulder_joint_position.z,
    ]

    # Calculate the vector from the shoulder joint position to the ground (downwards)
    shoulder_to_ground_vector = [0.0, -1.0, 0.0]

    # Calculate the angle between the two vectors
    # The angle is the arccos of the ratio of the dot product of the two vectors
    # and the product of the lengths of the two vectors
    arm_angle = get_angle_between_vectors(
        shoulder_to_element_vector, shoulder_to_ground_vector
    )

    return arm_angle

def get_arm_angle_deg(
    shoulder_joint_position: networking.element.Position,
    element: networking.element.Element,
):
    """
    Return the angle of the arm trying to reach the element in degrees.
    """
    # Calculate the arm angle
    arm_angle = get_arm_angle(shoulder_joint_position, element)

    # If the angle is infinity or undefined, return it
    if arm_angle == math.inf or arm_angle is None:
        return arm_angle

    return arm_angle * 180 / math.pi

def get_arm_ergonomics_cost(
    shoulder_joint_position: networking.element.Position,
    element: networking.element.Element,
):
    """
    Return the arm ergonomics cost of an element for a given shoulder joint position
    and arm length.

    The arm ergonomics cost is calculated as the angle between the vector
    from the shoulder joint position to the element and the vector from the
    shoulder joint position to the ground.
    """
    # Calculate the arm angle
    arm_angle = get_arm_angle(shoulder_joint_position, element)

    # If the angle is infinity, return 1
    if arm_angle == math.inf or arm_angle is None:
        return 1.0

    # Calculate the arm ergonomics cost normalized to [0, 1]
    arm_ergonomics_cost = arm_angle / math.pi / 2

    # Print arm angle in degrees
    # print(f"Arm angle: {arm_angle * 180 / math.pi}°")

    return arm_ergonomics_cost

def get_exp_arm_ergonomics_cost_from_angle(
    arm_angle: Union[float, None],
):
    """
    Return the exponential arm ergonomics cost of an element for a given arm angle in degrees.
    """
    # If the angle is infinity, return 1
    if arm_angle == math.inf or arm_angle is None:
        return 1.0

    # Calculate the exponential arm ergonomics cost normalized to [0, 1]
    # The cost grows exponentially with the angle
    # It has a maximum of 1 if the angle is 180°
    # It has a minimum of close to 0 if the angle is 0°
    # If the angle is greater than 180°, the cost wraps around and decreases
    # If the angle is less than 0°, the cost is calculated using the absolute value of the angle
    arm_angle_rad = arm_angle * np.pi / 180

    # Cost has formula: f(x) = { (e^(x mod 2π/π) - 1)/(e-1) , x mod 2π <= π
    #                          { (e^(-(x mod 2π-2π)/π) - 1)/(e-1) , x mod 2π > π
    # where x is the arm angle in radians
    x_mod = np.mod(arm_angle_rad, 2*np.pi)
    arm_ergonomics_cost = np.where(x_mod <= np.pi, (np.exp(x_mod/np.pi) - 1)/(np.exp(1)-1), 
                    (np.exp(-(x_mod-2*np.pi)/np.pi) - 1)/(np.exp(1)-1)).item()
    return arm_ergonomics_cost

def get_exp_arm_ergonomics_cost(
    shoulder_joint_position: networking.element.Position,
    element: networking.element.Element,
):
    """
    Return the exponential arm ergonomics cost of an element for a given shoulder joint position
    and arm length.

    The arm ergonomics cost is calculated as an exponential growth with the angle between the vector
    from the shoulder joint position to the element and the vector from the
    shoulder joint position to the ground.
    """
    # Calculate the arm angle in degrees
    arm_angle = get_arm_angle_deg(shoulder_joint_position, element)

    # Calculate cost
    arm_ergonomics_cost = get_exp_arm_ergonomics_cost_from_angle(arm_angle)
    return arm_ergonomics_cost


def get_hand_reachability_cost(
    hand_position: networking.element.Position,
    element: networking.element.Element,
    zone_thickness: float = 0.1,
    zone_cost_growth: float = 1.0,
    innermost_zone_cost: float = math.inf,
):
    """Return the hand reachability cost of an element.

    The cost is determined by the distance between the element and the
    hand position. There are zones as hollow spheres with a given thickness
    around the hand with different costs. The zone closest to the hand has a
    high cost, the next closest zone has the lowest cost. The cost increases
    stepwise between the zones. All zones are centered at the hand position
    and grow linearly in radius.
    """
    # Calculate the distance between the hand position and the element
    distance = math.sqrt(
        (hand_position.x - element.position.x) ** 2
        + (hand_position.y - element.position.y) ** 2
        + (hand_position.z - element.position.z) ** 2
    )

    # If the distance is smaller than the radius of the innermost zone,
    # the cost is +infinity
    if distance < zone_thickness:
        return innermost_zone_cost

    # Otherwise, the cost is the cost of the zone the element is currently in
    index_of_current_zone = math.floor(distance / zone_thickness)
    hand_reachability_cost = index_of_current_zone * zone_cost_growth

    return hand_reachability_cost


def evaluate_layouts(layouts: List[networking.layout.Layout]):
    # TODO: Implement cost evaluation at AUIT server
    # Placeholder for evaluating the costs: at-arms-length cost, neck ergonomics cost, and arm ergonomics cost
    # sum of the costs for each element in the layout
    # Define the eye position
    eye_position = networking.element.Position(x=0.0, y=0.0, z=0.0)

    # Define the shoulder joint position
    shoulder_joint_position = networking.element.Position(x=0.0, y=-1.0, z=0.0)
    # shoulder_joint_position = networking.element.Position(x=0.0, y=-2.0, z=0.0)  # for more realistic proportions

    # Define the arm length
    arm_length = 3.0
    # arm_length = 0.7  # for more realistic proportions

    # Define/calculate the hand position
    # The hand is at arm's length from the shoulder at a 45 degree angle
    hand_position = networking.element.Position(
        x=shoulder_joint_position.x + arm_length * np.cos(np.deg2rad(45)),
        y=shoulder_joint_position.y - arm_length * np.sin(np.deg2rad(45)),
        z=shoulder_joint_position.z,
    )

    # Objective costs
    costs = [
        [
            # sum(
            #     get_hand_reachability_cost(hand_position, element)
            #     for element in layout.items
            # ),  # hand reachability cost
            sum(
                get_arm_ergonomics_cost(shoulder_joint_position, element)
                for element in layout.items
            ),  # arm ergonomics cost
            sum(
                get_neck_ergonomics_cost(eye_position, element)
                for element in layout.items
            ),  # neck ergonomics cost
        ]
        for layout in layouts
    ]

    # Constraint violations
    violations = [
        [
            sum(
                get_at_arms_length_cost(shoulder_joint_position, arm_length, element)
                for element in layout.items
            ),  # at-arms-length cost
        ]
        for layout in layouts
    ]

    return costs, violations


def main():
    """Main function."""
    # Spin up a server for the layout evaluation in a separate thread
    def run_server():
        return server.run_server(port=5556)

    server_thread = threading.Thread(target=run_server)

    # Start the server thread
    server_thread.start()

    # Set the number of objectives, constraints and the initial layout
    n_objectives = 2
    n_constraints = 1
    layout = networking.layout.Layout(
        items=[
            networking.element.Element(
                id="1",
                position=networking.element.Position(x=14, y=2, z=7),
                rotation=networking.element.Rotation(x=0.4, y=0.1, z=0.9, w=1)
            )
        ]
    )

    # Get Pareto optimal solutions from the solver
    solutions = get_pareto_optimal_solutions(
        n_objectives, n_constraints, layout
    )

    # Print the solutions (placeholder for displaying the solutions in 3D)
    print("Pareto optimal solutions:")
    for solution in solutions:
        print(solution)


if __name__ == "__main__":
    main()
