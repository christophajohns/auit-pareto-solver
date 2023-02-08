"""Test functions for AUIT.py"""

# Load the AUIT module
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

# from .. import AUIT  # Uncomment this line for docstring hints

import AUIT  # Uncomment this line for running the tests

# Hyperparameters
EYE_POSITION = AUIT.networking.element.Position(x=0.0, y=0.0, z=0.0)
SHOULDER_JOINT_POSITION = AUIT.networking.element.Position(x=0.0, y=-0.3, z=0.0)
ARM_LENGTH = 3.0
USER_HEIGHT = 1.80
outstretched_arm_hand_position = AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x + ARM_LENGTH, y=SHOULDER_JOINT_POSITION.y, z=SHOULDER_JOINT_POSITION.z)
resting_hand_position = AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x, y=SHOULDER_JOINT_POSITION.y - ARM_LENGTH, z=SHOULDER_JOINT_POSITION.z)

def get_element_at_eye_level():
    """Returns a UI element at eye level."""
    # Calculate the x-position of the element at eye level
    # in arm's length from the shoulder joint
    y_distance_from_eye_to_shoulder = abs(EYE_POSITION.y - SHOULDER_JOINT_POSITION.y)
    element_x_position = (ARM_LENGTH**2 - y_distance_from_eye_to_shoulder**2) ** 0.5
    # print("Element x position: {}".format(element_x_position))

    # Define test element for cost evaluation at eye level
    element_at_eye_level = AUIT.networking.element.Element(
        id="test_element",
        position=AUIT.networking.element.Position(x=element_x_position, y=EYE_POSITION.y, z=0.0),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_at_eye_level

def get_element_at_waist_level():
    """Returns a UI element at waist level."""
    # Define test element for cost evaluation at waist level
    element_at_waist_level = AUIT.networking.element.Element(
        id="test_element",
        position=AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x, y=SHOULDER_JOINT_POSITION.y - ARM_LENGTH, z=SHOULDER_JOINT_POSITION.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_at_waist_level

def get_element_at_arms_length():
    """Returns a UI element in front of the shoulder joint at arm's length."""
    # Define test element for cost evaluation at arm's length
    element_at_arms_length = AUIT.networking.element.Element(
        id="test_element",
        position=AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x + ARM_LENGTH, y=SHOULDER_JOINT_POSITION.y, z=SHOULDER_JOINT_POSITION.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_at_arms_length

def get_element_not_at_arms_length():
    """Returns a UI element in front of the shoulder joint at arm's length."""
    # Define test element for cost evaluation not at arm's length
    element_not_at_arms_length = AUIT.networking.element.Element(
        id="test_element_not_at_arms_length",
        position=AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x + ARM_LENGTH * 1.5, y=SHOULDER_JOINT_POSITION.y, z=SHOULDER_JOINT_POSITION.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_not_at_arms_length

def get_element_at_hand_position():
    """Returns a UI element at the hand position."""
    # Define test element for cost evaluation at the hand position
    element_at_hand_position = AUIT.networking.element.Element(
        id="test_element",
        position=outstretched_arm_hand_position,
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_at_hand_position

def get_element_in_innermost_zone():
    """Returns a UI element in the innermost zone distance to the hand."""
    # Define test element for cost evaluation in the innermost zone of
    # the hand reachability (i.e., within 0.1 m)
    element_in_innermost_zone = AUIT.networking.element.Element(
        id="test_element_in_innermost_zone",
        position=AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x + ARM_LENGTH - 0.09, y=SHOULDER_JOINT_POSITION.y, z=SHOULDER_JOINT_POSITION.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_in_innermost_zone

def get_element_in_first_zone():
    """Returns a UI element in the first zone distance to the hand."""
    # Define test element for cost evaluation in the first zone outside
    # the innermost zone of the hand reachability (i.e., within 0.2 m)
    element_in_first_zone = AUIT.networking.element.Element(
        id="test_element_in_first_zone",
        position=AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x + ARM_LENGTH - 0.19, y=SHOULDER_JOINT_POSITION.y, z=SHOULDER_JOINT_POSITION.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_in_first_zone

def get_element_in_second_zone():
    """Returns a UI element in the second zone distance to the hand."""
    # Define test element for cost evaluation in the second zone outside
    # the innermost zone of the hand reachability (i.e., within 0.3 m)
    element_in_second_zone = AUIT.networking.element.Element(
        id="test_element_in_second_zone",
        position=AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x + ARM_LENGTH - 0.29, y=SHOULDER_JOINT_POSITION.y, z=SHOULDER_JOINT_POSITION.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_in_second_zone

def get_element_at_ground_level():
    """Returns a UI element at ground level."""
    # Define test element for cost evaluation at ground level
    element_at_ground_level = AUIT.networking.element.Element(
        id="test_element",
        position=AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x, y=SHOULDER_JOINT_POSITION.y - USER_HEIGHT, z=SHOULDER_JOINT_POSITION.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_at_ground_level

def nearly_equal(float_1: float, float_2: float, tolerance: float = 1e-4):
    """Returns whether two floats are equal respective the given tolerance."""
    return abs(float_2 - float_1) < tolerance

def test_cost_evaluation_at_eye_level():
    """Test cost evaluation for an element at eye level."""
    print("Testing cost evaluation at eye level...")

    # Define test element for cost evaluation at eye level
    element_at_eye_level = get_element_at_eye_level()

    # Define tolerance
    tolerance = 0.0

    # Calculate "at arm's length" reachability cost
    at_arms_length_cost = AUIT.get_at_arms_length_cost(
        SHOULDER_JOINT_POSITION, ARM_LENGTH, element_at_eye_level, tolerance
    )

    # Check the cost
    # Define tolerance for floating point comparison
    assert (
        nearly_equal(at_arms_length_cost, 0)
    ), "'At arm's length' reachability cost should be 0. Got: {}".format(
        at_arms_length_cost
    )

    # Calculate neck ergonomics cost
    neck_ergonomics_cost = AUIT.get_neck_ergonomics_cost(
        EYE_POSITION, element_at_eye_level
    )

    # Check the cost
    assert (
        neck_ergonomics_cost == 0
    ), "Neck ergonomics cost should be 0. Got: {}".format(neck_ergonomics_cost)
    print("Neck ergonomics cost: {}".format(neck_ergonomics_cost))

    # Calculate arm ergonomics cost
    arm_ergonomics_cost = AUIT.get_arm_ergonomics_cost(
        SHOULDER_JOINT_POSITION, element_at_eye_level
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

    # Define test element for cost evaluation at waist level
    element_at_waist_level = get_element_at_waist_level()

    # Define tolerance
    tolerance = 0.001

    # Calculate "at arm's length" reachability cost
    at_arms_length_cost = AUIT.get_at_arms_length_cost(
        SHOULDER_JOINT_POSITION, ARM_LENGTH, element_at_waist_level, tolerance
    )

    # Check the cost
    assert (
        nearly_equal(at_arms_length_cost, 0, tolerance=1e-2)
    ), "'At arm's length' reachability cost should be 0. Got: {}".format(
        at_arms_length_cost
    )

    # Calculate neck ergonomics cost
    neck_ergonomics_cost = AUIT.get_neck_ergonomics_cost(
        EYE_POSITION, element_at_waist_level
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
        SHOULDER_JOINT_POSITION, element_at_waist_level
    )

    # Check the cost
    assert arm_ergonomics_cost == 0, "Arm ergonomics cost should be 0. Got: {}".format(
        arm_ergonomics_cost
    )
    print("Arm ergonomics cost: {}".format(arm_ergonomics_cost))


def test_cost_evaluation_at_arms_length():
    """Test cost evaluation for an element at arm's length."""
    # Define test element for cost evaluation at arm's length
    element_at_arms_length = get_element_at_arms_length()

    # Define tolerance
    tolerance = 0.001

    # Calculate "at arm's length" reachability cost
    at_arms_length_cost = AUIT.get_at_arms_length_cost(
        SHOULDER_JOINT_POSITION, ARM_LENGTH, element_at_arms_length, tolerance
    )

    # Check the cost
    assert (
        nearly_equal(at_arms_length_cost, 0, tolerance=1e-2)
    ), "'At arm's length' reachability cost should be 0. Got: {}".format(
        at_arms_length_cost
    )

    # Define test element for cost evaluation not at arm's length
    element_not_at_arms_length = get_element_not_at_arms_length()

    # Calculate "at arm's length" reachability cost
    at_arms_length_cost = AUIT.get_at_arms_length_cost(
        SHOULDER_JOINT_POSITION, ARM_LENGTH, element_not_at_arms_length
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

    # Define test element for cost evaluation at the hand position
    element_at_hand_position = get_element_at_hand_position()

    # Calculate hand reachability cost
    hand_reachability_cost = AUIT.get_hand_reachability_cost(
        outstretched_arm_hand_position, element_at_hand_position
    )

    # Check the cost
    assert hand_reachability_cost == float(
        "inf"
    ), "Hand reachability cost should be +inf. Got: {}".format(hand_reachability_cost)

    # Define test element for cost evaluation in the innermost zone of
    # the hand reachability (i.e., within 0.1 m)
    element_in_innermost_zone = get_element_in_innermost_zone()

    # Calculate hand reachability cost
    hand_reachability_cost = AUIT.get_hand_reachability_cost(
        outstretched_arm_hand_position, element_in_innermost_zone
    )

    # Check the cost (should be +inf)
    assert hand_reachability_cost == float(
        "inf"
    ), "Hand reachability cost should be +inf. Got: {}".format(hand_reachability_cost)

    # Define test element for cost evaluation in the first zone outside
    # the innermost zone of the hand reachability (i.e., within 0.2 m)
    element_in_first_zone = get_element_in_first_zone()

    # Calculate hand reachability cost
    hand_reachability_cost = AUIT.get_hand_reachability_cost(
        outstretched_arm_hand_position, element_in_first_zone
    )

    # Check the cost (should be 1.0)
    assert (
        hand_reachability_cost == 1.0
    ), "Hand reachability cost should be 1.0. Got: {}".format(hand_reachability_cost)

    # Define test element for cost evaluation in the second zone outside
    # the innermost zone of the hand reachability (i.e., within 0.3 m)
    element_in_second_zone = get_element_in_second_zone()

    # Calculate hand reachability cost
    hand_reachability_cost = AUIT.get_hand_reachability_cost(
        outstretched_arm_hand_position, element_in_second_zone
    )

    # Check the cost (should be 2.0)
    assert (
        hand_reachability_cost == 2.0
    ), "Hand reachability cost should be 2.0. Got: {}".format(hand_reachability_cost)

def test_torso_ergonomics_cost():
    """Test torso ergonomics cost."""
    print("Testing torso ergonomics cost...")

    # Get element at waist level
    element_at_waist_level = get_element_at_waist_level()
    element_at_waist_level.position.x += 0.01 # Move element slightly away from waist

    # Calculate torso ergonomics cost
    torso_ergonomics_cost_at_waist_level = AUIT.get_torso_ergonomics_cost(
        waist_position=resting_hand_position, element=element_at_waist_level, verbose=False
    )

    # Check the cost
    assert nearly_equal(torso_ergonomics_cost_at_waist_level, 0), "Torso ergonomics at waist level cost should be 0. Got: {}".format(torso_ergonomics_cost_at_waist_level)

    # Get element at shoulder level
    element_at_shoulder_level = get_element_at_arms_length()

    # Calculate torso ergonomics cost
    torso_ergonomics_cost_at_shoulder_level = AUIT.get_torso_ergonomics_cost(
        waist_position=resting_hand_position, element=element_at_shoulder_level, verbose=False
    )

    # Check the cost
    assert nearly_equal(torso_ergonomics_cost_at_shoulder_level, 0), "Torso ergonomics at shoulder level cost should be 0. Got: {}".format(torso_ergonomics_cost_at_shoulder_level)

    # Get element at ground level
    element_at_ground_level = get_element_at_ground_level()

    # Calculate torso ergonomics cost
    torso_ergonomics_cost_at_ground_level = AUIT.get_torso_ergonomics_cost(
        waist_position=resting_hand_position, element=element_at_ground_level, verbose=False
    )

    # Check the cost
    assert nearly_equal(torso_ergonomics_cost_at_ground_level, 0), "Torso ergonomics at ground level cost should be > 0. Got: {}".format(torso_ergonomics_cost_at_ground_level)



def test_cost_evaluation():
    """Test cost evaluation."""
    test_torso_ergonomics_cost()
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
