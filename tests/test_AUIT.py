"""Test functions for AUIT.py"""

# Load the AUIT module
import sys
import os
import math

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

# from .. import AUIT  # Uncomment this line for docstring hints

import AUIT  # Uncomment this line for running the tests


def test_cost_evaluation_at_eye_level():
    """Test cost evaluation for an element at eye level."""
    print("Testing cost evaluation at eye level...")
    # Define the eye position
    eye_position = AUIT.networking.element.Position(x=0.0, y=0.0, z=0.0)

    # Define the shoulder joint position
    shoulder_joint_position = AUIT.networking.element.Position(x=0.0, y=-1.0, z=0.0)

    # Define the arm length
    arm_length = 3.0

    # Calculate the x-position of the element at eye level
    # in arm's length from the shoulder joint
    y_distance_from_eye_to_shoulder = abs(eye_position.y - shoulder_joint_position.y)
    element_x_position = (arm_length**2 - y_distance_from_eye_to_shoulder**2) ** 0.5
    print("Element x position: {}".format(element_x_position))

    # Define test element for cost evaluation at eye level
    element_at_eye_level = AUIT.networking.element.Element(
        id="test_element",
        position=AUIT.networking.element.Position(x=element_x_position, y=eye_position.y, z=0.0),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    # Define tolerance
    tolerance = 0.0

    # Calculate "at arm's length" reachability cost
    at_arms_length_cost = AUIT.get_at_arms_length_cost(
        shoulder_joint_position, arm_length, element_at_eye_level, tolerance
    )

    # Check the cost
    # Define tolerance for floating point comparison
    float_tolerance = 0.0001
    assert (
        at_arms_length_cost < float_tolerance
    ), "'At arm's length' reachability cost should be 0. Got: {}".format(
        at_arms_length_cost
    )

    # Calculate neck ergonomics cost
    neck_ergonomics_cost = AUIT.get_neck_ergonomics_cost(
        eye_position, element_at_eye_level
    )

    # Check the cost
    assert (
        neck_ergonomics_cost == 0
    ), "Neck ergonomics cost should be 0. Got: {}".format(neck_ergonomics_cost)
    print("Neck ergonomics cost: {}".format(neck_ergonomics_cost))

    # Calculate arm ergonomics cost
    arm_ergonomics_cost = AUIT.get_arm_ergonomics_cost(
        shoulder_joint_position, element_at_eye_level
    )

    # Check the cost
    assert (
        arm_ergonomics_cost > 0
    ), "Arm ergonomics cost should be greater than 0. Got: {}".format(
        arm_ergonomics_cost
    )
    print("Arm ergonomics cost: {}".format(arm_ergonomics_cost))


def test_cost_evaluation_at_waist_level():
    """Test cost evaluation for an element at the waist."""
    print("Testing cost evaluation at waist level...")

    # Define the eye position
    eye_position = AUIT.networking.element.Position(x=0.0, y=0.0, z=0.0)

    # Define the shoulder joint position
    shoulder_joint_position = AUIT.networking.element.Position(x=0.0, y=-1.0, z=0.0)

    # Define the arm length
    arm_length = 3.0

    # Define test element for cost evaluation at waist level
    element_at_waist_level = AUIT.networking.element.Element(
        id="test_element",
        position=AUIT.networking.element.Position(x=shoulder_joint_position.x, y=shoulder_joint_position.y - arm_length, z=shoulder_joint_position.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    # Define tolerance
    tolerance = 0.001

    # Calculate "at arm's length" reachability cost
    at_arms_length_cost = AUIT.get_at_arms_length_cost(
        shoulder_joint_position, arm_length, element_at_waist_level, tolerance
    )

    # Check the cost
    # Define tolerance for floating point comparison
    float_tolerance = 0.0001
    assert (
        at_arms_length_cost < float_tolerance
    ), "'At arm's length' reachability cost should be 0. Got: {}".format(
        at_arms_length_cost
    )

    # Calculate neck ergonomics cost
    neck_ergonomics_cost = AUIT.get_neck_ergonomics_cost(
        eye_position, element_at_waist_level
    )

    # Check the cost
    assert (
        neck_ergonomics_cost > 0
    ), "Neck ergonomics cost should be greater than 0. Got: {}".format(
        neck_ergonomics_cost
    )
    print("Neck ergonomics cost: {}".format(neck_ergonomics_cost))

    # Calculate arm ergonomics cost
    arm_ergonomics_cost = AUIT.get_arm_ergonomics_cost(
        shoulder_joint_position, element_at_waist_level
    )

    # Check the cost
    assert arm_ergonomics_cost == 0, "Arm ergonomics cost should be 0. Got: {}".format(
        arm_ergonomics_cost
    )
    print("Arm ergonomics cost: {}".format(arm_ergonomics_cost))


def test_cost_evaluation_at_arms_length():
    """Test cost evaluation for an element at arm's length."""
    # Define the shoulder joint position
    shoulder_joint_position = AUIT.networking.element.Position(x=0.0, y=-1.0, z=0.0)

    # Define the arm length
    arm_length = 3.0

    # Define test element for cost evaluation at arm's length
    element_at_arms_length = AUIT.networking.element.Element(
        id="test_element",
        position=AUIT.networking.element.Position(x=shoulder_joint_position.x + arm_length, y=shoulder_joint_position.y, z=shoulder_joint_position.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    # Define tolerance
    tolerance = 0.001

    # Calculate "at arm's length" reachability cost
    at_arms_length_cost = AUIT.get_at_arms_length_cost(
        shoulder_joint_position, arm_length, element_at_arms_length, tolerance
    )

    # Check the cost
    # Define tolerance for floating point comparison
    float_tolerance = 0.0001
    assert (
        at_arms_length_cost < float_tolerance
    ), "'At arm's length' reachability cost should be 0. Got: {}".format(
        at_arms_length_cost
    )

    # Define test element for cost evaluation not at arm's length
    element_not_at_arms_length = AUIT.networking.element.Element(
        id="test_element_not_at_arms_length",
        position=AUIT.networking.element.Position(x=shoulder_joint_position.x + arm_length * 1.5, y=shoulder_joint_position.y, z=shoulder_joint_position.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    # Calculate "at arm's length" reachability cost
    at_arms_length_cost = AUIT.get_at_arms_length_cost(
        shoulder_joint_position, arm_length, element_not_at_arms_length
    )

    # Check the cost
    assert (
        at_arms_length_cost > 0
    ), "'At arm's length' reachability cost should be greater than 0. Got: {}".format(
        at_arms_length_cost
    )


def test_hand_reachability_cost():
    """Test hand reachability cost."""
    print("Testing hand reachability cost...")

    # Define the shoulder joint position
    shoulder_joint_position = AUIT.networking.element.Position(x=0.0, y=-1.0, z=0.0)

    # Define the arm length
    arm_length = 3.0

    # Define the hand position
    hand_position = AUIT.networking.element.Position(x=shoulder_joint_position.x + arm_length, y=shoulder_joint_position.y, z=shoulder_joint_position.z)

    # Define test element for cost evaluation at the hand position
    element_at_hand_position = AUIT.networking.element.Element(
        id="test_element",
        position=hand_position,
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    # Calculate hand reachability cost
    hand_reachability_cost = AUIT.get_hand_reachability_cost(
        hand_position, element_at_hand_position
    )

    # Check the cost
    assert hand_reachability_cost == float(
        "inf"
    ), "Hand reachability cost should be +inf. Got: {}".format(hand_reachability_cost)

    # Define test element for cost evaluation in the innermost zone of
    # the hand reachability (i.e., within 0.1 m)
    element_in_innermost_zone = AUIT.networking.element.Element(
        id="test_element_in_innermost_zone",
        position=AUIT.networking.element.Position(x=shoulder_joint_position.x + arm_length - 0.09, y=shoulder_joint_position.y, z=shoulder_joint_position.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    # Calculate hand reachability cost
    hand_reachability_cost = AUIT.get_hand_reachability_cost(
        hand_position, element_in_innermost_zone
    )

    # Check the cost (should be +inf)
    assert hand_reachability_cost == float(
        "inf"
    ), "Hand reachability cost should be +inf. Got: {}".format(hand_reachability_cost)

    # Define test element for cost evaluation in the first zone outside
    # the innermost zone of the hand reachability (i.e., within 0.2 m)
    element_in_first_zone = AUIT.networking.element.Element(
        id="test_element_in_first_zone",
        position=AUIT.networking.element.Position(x=shoulder_joint_position.x + arm_length - 0.19, y=shoulder_joint_position.y, z=shoulder_joint_position.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    # Calculate hand reachability cost
    hand_reachability_cost = AUIT.get_hand_reachability_cost(
        hand_position, element_in_first_zone
    )

    # Check the cost (should be 1.0)
    assert (
        hand_reachability_cost == 1.0
    ), "Hand reachability cost should be 1.0. Got: {}".format(hand_reachability_cost)

    # Define test element for cost evaluation in the second zone outside
    # the innermost zone of the hand reachability (i.e., within 0.3 m)
    element_in_second_zone = AUIT.networking.element.Element(
        id="test_element_in_second_zone",
        position=AUIT.networking.element.Position(x=shoulder_joint_position.x + arm_length - 0.29, y=shoulder_joint_position.y, z=shoulder_joint_position.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    # Calculate hand reachability cost
    hand_reachability_cost = AUIT.get_hand_reachability_cost(
        hand_position, element_in_second_zone
    )

    # Check the cost (should be 2.0)
    assert (
        hand_reachability_cost == 2.0
    ), "Hand reachability cost should be 2.0. Got: {}".format(hand_reachability_cost)


def test_cost_evaluation():
    """Test cost evaluation."""
    test_hand_reachability_cost()
    test_cost_evaluation_at_eye_level()
    test_cost_evaluation_at_waist_level()
    test_cost_evaluation_at_arms_length()


def test_AUIT():
    """Test AUIT.py."""
    test_cost_evaluation()


def main():
    """Main function."""
    test_AUIT()


if __name__ == "__main__":
    main()
