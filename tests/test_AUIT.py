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

def get_element_not_at_arms_length(distance_factor=1.5):
    """Returns a UI element in front of the shoulder joint not at arm's length."""
    # Define test element for cost evaluation not at arm's length
    element_not_at_arms_length = AUIT.networking.element.Element(
        id="test_element_not_at_arms_length",
        position=AUIT.networking.element.Position(x=SHOULDER_JOINT_POSITION.x + ARM_LENGTH * distance_factor, y=SHOULDER_JOINT_POSITION.y, z=SHOULDER_JOINT_POSITION.z),
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return element_not_at_arms_length

def get_element_far_from_arms_length():
    """Returns a UI element in front of the shoulder joint far from arm's length."""
    # Define test element for cost evaluation far from arm's length
    element_far_from_arms_length = get_element_not_at_arms_length(distance_factor=30)
    return element_far_from_arms_length

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

    # Calculate exponential arm ergonomics cost
    exp_arm_ergonomics_cost = AUIT.get_exp_arm_ergonomics_cost(
        SHOULDER_JOINT_POSITION, element_at_eye_level
    )

    test_exp_arm_ergonomics_cost()

    # Check the cost
    assert (
        exp_arm_ergonomics_cost > 0
    ), "Exponential arm ergonomics cost should be greater than 0. Got: {}".format(
        exp_arm_ergonomics_cost
    )
    print("Arm ergonomics cost: {}".format(exp_arm_ergonomics_cost))
    assert (
        arm_ergonomics_cost < exp_arm_ergonomics_cost
    ), "Arm ergonomics cost should be smaller than exponential arm ergonomics cost. Got: {} and {}".format(
        arm_ergonomics_cost, exp_arm_ergonomics_cost
    )

def test_exp_arm_ergonomics_cost():
    """Test exponential arm ergonomics cost."""
    print("Testing exponential arm ergonomics cost...")

    assert (
        nearly_equal(AUIT.get_exp_arm_ergonomics_cost_from_angle(0), 0)
    ), "Exponential arm ergonomics cost should be 0 for angle 0. Got: {}".format(
        AUIT.get_exp_arm_ergonomics_cost_from_angle(0)
    )

    assert (
        nearly_equal(AUIT.get_exp_arm_ergonomics_cost_from_angle(180), 1)
    ), "Exponential arm ergonomics cost should be 1 for angle 180. Got: {}".format(
        AUIT.get_exp_arm_ergonomics_cost_from_angle(180)
    )

    assert (
        nearly_equal(AUIT.get_exp_arm_ergonomics_cost_from_angle(185), AUIT.get_exp_arm_ergonomics_cost_from_angle(175))
    ), "Exponential arm ergonomics cost should be equal for angles 185 and 175. Got: {} and {}".format(
        AUIT.get_exp_arm_ergonomics_cost_from_angle(185), AUIT.get_exp_arm_ergonomics_cost_from_angle(175)
    )

    assert (
        nearly_equal(AUIT.get_exp_arm_ergonomics_cost_from_angle(-10), AUIT.get_exp_arm_ergonomics_cost_from_angle(10))
    ), "Exponential arm ergonomics cost should be equal for angles -10 and 10. Got: {} and {}".format(
        AUIT.get_exp_arm_ergonomics_cost_from_angle(-10), AUIT.get_exp_arm_ergonomics_cost_from_angle(10)
    )

    assert (
        nearly_equal(AUIT.get_exp_arm_ergonomics_cost_from_angle(0), AUIT.get_exp_arm_ergonomics_cost_from_angle(360))
    ), "Exponential arm ergonomics cost should be equal for angles 0 and 360. Got: {} and {}".format(
        AUIT.get_exp_arm_ergonomics_cost_from_angle(0), AUIT.get_exp_arm_ergonomics_cost_from_angle(360)
    )

    assert (
        AUIT.get_exp_arm_ergonomics_cost_from_angle(90) > AUIT.get_exp_arm_ergonomics_cost_from_angle(0)
    ), "Exponential arm ergonomics cost should be greater for angle 90 than for angle 0. Got: {} and {}".format(
        AUIT.get_exp_arm_ergonomics_cost_from_angle(90), AUIT.get_exp_arm_ergonomics_cost_from_angle(0)
    )

    assert (
        AUIT.get_exp_arm_ergonomics_cost_from_angle(180) > AUIT.get_exp_arm_ergonomics_cost_from_angle(90)
    ), "Exponential arm ergonomics cost should be greater for angle 180 than for angle 90. Got: {} and {}".format(
        AUIT.get_exp_arm_ergonomics_cost_from_angle(180), AUIT.get_exp_arm_ergonomics_cost_from_angle(90)
    )

    assert (
        AUIT.get_exp_arm_ergonomics_cost_from_angle(270) < AUIT.get_exp_arm_ergonomics_cost_from_angle(180)
    ), "Exponential arm ergonomics cost should be less for angle 270 than for angle 180. Got: {} and {}".format(
        AUIT.get_exp_arm_ergonomics_cost_from_angle(270), AUIT.get_exp_arm_ergonomics_cost_from_angle(180)
    )

    assert (
        nearly_equal(AUIT.get_exp_arm_ergonomics_cost_from_angle(370), AUIT.get_exp_arm_ergonomics_cost_from_angle(10))
    ), "Exponential arm ergonomics cost should be equal for angles 370 and 10. Got: {} and {}".format(
        AUIT.get_exp_arm_ergonomics_cost_from_angle(370), AUIT.get_exp_arm_ergonomics_cost_from_angle(10)
    )

    # Test that cost grows quasi-exponentially for angles between 0 and 180 by checking that the difference between
    # the costs for angles growing by 10 degrees grows
    initial_diff_of_costs = AUIT.get_exp_arm_ergonomics_cost_from_angle(10) - AUIT.get_exp_arm_ergonomics_cost_from_angle(0)
    for i in range(1, 17):
        diff_of_costs = AUIT.get_exp_arm_ergonomics_cost_from_angle(10 * (i + 1)) - AUIT.get_exp_arm_ergonomics_cost_from_angle(10 * i)
        assert (
            diff_of_costs > initial_diff_of_costs
        ), "Difference of exponential arm ergonomics costs should grow for angles growing by 10 degrees. Got: {} and {}".format(
            initial_diff_of_costs, diff_of_costs
        )
        initial_diff_of_costs = diff_of_costs

    # Test that cost decreases quasi-exponentially for angles between 180 and 360 by checking that the difference between
    # the costs for angles growing by 10 degrees decreases
    initial_diff_of_costs = AUIT.get_exp_arm_ergonomics_cost_from_angle(190) - AUIT.get_exp_arm_ergonomics_cost_from_angle(180)
    for i in range(1, 17):
        diff_of_costs = AUIT.get_exp_arm_ergonomics_cost_from_angle(10 * (i + 1) + 180) - AUIT.get_exp_arm_ergonomics_cost_from_angle(10 * i + 180)
        assert (
            diff_of_costs > initial_diff_of_costs
        ), "Difference of exponential arm ergonomics costs should decrease for angles growing by 10 degrees. Got: {} and {}".format(
            initial_diff_of_costs, diff_of_costs
        )
        initial_diff_of_costs = diff_of_costs
    
    # Test that the cost increases again quasi-exponentially for angles between 360 and 540 by checking that the difference between
    # the costs for angles growing by 10 degrees grows
    initial_diff_of_costs = AUIT.get_exp_arm_ergonomics_cost_from_angle(370) - AUIT.get_exp_arm_ergonomics_cost_from_angle(360)
    for i in range(1, 17):
        diff_of_costs = AUIT.get_exp_arm_ergonomics_cost_from_angle(10 * (i + 1) + 360) - AUIT.get_exp_arm_ergonomics_cost_from_angle(10 * i + 360)
        assert (
            diff_of_costs > initial_diff_of_costs
        ), "Difference of exponential arm ergonomics costs should grow for angles growing by 10 degrees. Got: {} and {}".format(
            initial_diff_of_costs, diff_of_costs
        )
        initial_diff_of_costs = diff_of_costs

    # Test that the cost decreases again quasi-exponentially for angles between 540 and 720 by checking that the difference between
    # the costs for angles growing by 10 degrees decreases
    initial_diff_of_costs = AUIT.get_exp_arm_ergonomics_cost_from_angle(550) - AUIT.get_exp_arm_ergonomics_cost_from_angle(540)
    for i in range(1, 17):
        diff_of_costs = AUIT.get_exp_arm_ergonomics_cost_from_angle(10 * (i + 1) + 540) - AUIT.get_exp_arm_ergonomics_cost_from_angle(10 * i + 540)
        assert (
            diff_of_costs > initial_diff_of_costs
        ), "Difference of exponential arm ergonomics costs should decrease for angles growing by 10 degrees. Got: {} and {}".format(
            initial_diff_of_costs, diff_of_costs
        )
        initial_diff_of_costs = diff_of_costs


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

    # Define test element for cost evaluation far from arm's length
    element_far_from_arms_length = get_element_far_from_arms_length()

    # Calculate "at arm's length" reachability cost
    at_arms_length_cost = AUIT.get_at_arms_length_cost(
        SHOULDER_JOINT_POSITION, ARM_LENGTH, element_far_from_arms_length
    )

    # Check the cost
    assert (
        at_arms_length_cost > 0
    ), "'At arm's length' reachability cost should be greater than 0. Got: {}".format(
        at_arms_length_cost
    )
    # Check the cost
    assert (
        at_arms_length_cost < 1
    ), "'At arm's length' reachability cost should be less than 1. Got: {}".format(
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

def test_semantic_cost():
    """Test the semantic cost (i.e., a cost that is based on the distance
    between the closest semantically related object in the user's environment
    and the element as weighted by the association score)."""
    print("Testing semantic cost...")

    # Define an association dictionary that holds information about the
    # semantic association between the element and the objects in the user's
    # environment as well as about the position of the objects

    # Test with a single association
    association_dict = {
        "objects": [
            {
                "semantic_class": "headphones",
                "position": AUIT.networking.element.Position(x=-0.4, y=-0.4, z=0.3),
                "positive_association_score": 1,
                "negative_association_score": 0,
            },
        ]
    }
    element = AUIT.networking.element.Element(
        id="test_element",
        position=association_dict["objects"][0]["position"],
        rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    semantic_cost = AUIT.get_semantic_cost(
        element=element, association_dict=association_dict
    )
    assert nearly_equal(semantic_cost, 0), "Semantic cost should be close to 0. Got: {}".format(semantic_cost)

    # Test with two associations
    association_dict_two_assoc = {
        "objects": [
            {
                "semantic_class": "headphones",
                "position": AUIT.networking.element.Position(x=-0.4, y=-0.4, z=0.3),
                "positive_association_score": 1,
                "negative_association_score": 0,
            },
            {
                "semantic_class": "display",
                "position": AUIT.networking.element.Position(x=0, y=0, z=0.8),
                "positive_association_score": 0.7,
                "negative_association_score": 0.2,
            }
        ]
    }
    semantic_cost_two_associations = AUIT.get_semantic_cost(
        element=element, association_dict=association_dict_two_assoc
    )
    assert nearly_equal(semantic_cost_two_associations, 0), "Semantic cost should be close to 0. Got: {}".format(semantic_cost_two_associations)

    # Move the element slightly away from the object but use only one association
    element.position.x += 0.2
    semantic_cost_slightly_away = AUIT.get_semantic_cost(
        element=element, association_dict=association_dict
    )
    assert nearly_equal(semantic_cost_slightly_away, 0), "Semantic cost should be close to 0. Got: {}".format(semantic_cost_slightly_away)

    
    # Check the cost away from the headphones and closer to the display
    semantic_cost_two_associations_slightly_away = AUIT.get_semantic_cost(
        element=element, association_dict=association_dict_two_assoc
    )
    assert semantic_cost_two_associations_slightly_away > semantic_cost_two_associations, "Semantic cost should be > than before. Got: {}, Before: {}".format(semantic_cost_two_associations_slightly_away, semantic_cost_two_associations)

    # Flip the headphones association score
    association_dict = {
        "objects": [
            {
                "semantic_class": "headphones",
                "position": AUIT.networking.element.Position(x=-0.4, y=-0.4, z=0.3),
                "positive_association_score": 0,
                "negative_association_score": 1,
            },
            {
                "semantic_class": "display",
                "position": AUIT.networking.element.Position(x=0, y=0, z=0.8),
                "positive_association_score": 0.7,
                "negative_association_score": 0.2,
            }
        ]
    }
    semantic_cost_two_associations_flipped = AUIT.get_semantic_cost(
        element=element, association_dict=association_dict
    )
    assert semantic_cost_two_associations_flipped > semantic_cost_two_associations, "Semantic cost should be < than before. Got: {}, Before: {}".format(semantic_cost_two_associations_flipped, semantic_cost_two_associations)

    # Move element to display
    element.position = association_dict["objects"][1]["position"]
    semantic_cost_two_associations_at_display = AUIT.get_semantic_cost(
        element=element, association_dict=association_dict
    )
    assert semantic_cost_two_associations_at_display < semantic_cost_two_associations_flipped, "Semantic cost should be < than before. Got: {}, Before: {}".format(semantic_cost_two_associations_at_display, semantic_cost_two_associations_flipped)

    association_dict = {
        "objects": [
            {
                "semantic_class": "headphones",
                "position": AUIT.networking.element.Position(x=-0.4, y=-0.4, z=0.3),
                "positive_association_score": 1.0,
                "negative_association_score": 0,
            },
            {
                "semantic_class": "display",
                "position": AUIT.networking.element.Position(x=0, y=0, z=0.8),
                "positive_association_score": .5,
                "negative_association_score": 0.2,
            },
            {
                "semantic_class": "clock",
                "position": AUIT.networking.element.Position(x=-0, y=-1, z=1),
                "positive_association_score": 0,
                "negative_association_score": 0,
            },
            {
                "semantic_class": "glasses",
                "position": AUIT.networking.element.Position(x=0.4, y=-0.4, z=0.2),
                "positive_association_score": .1,
                "negative_association_score": 1.0,
            },
        ]
    }

    # Define an element that is positioned at the most relevant object
    for obj in association_dict["objects"]:
        element = AUIT.networking.element.Element(
            id="test_element",
            position=obj["position"],
            rotation=AUIT.networking.element.Rotation(x=0.0, y=0.0, z=0.0, w=1.0),
        )

        # Calculate semantic cost
        semantic_cost = AUIT.get_semantic_cost(
            element=element, association_dict=association_dict
        )

        print("Semantic cost for element at position {} ({}): {}".format(
            element.position, obj["semantic_class"], semantic_cost
        ))

def test_cost_evaluation():
    """Test cost evaluation."""
    test_semantic_cost()
    # test_torso_ergonomics_cost()
    # test_hand_reachability_cost()
    # test_cost_evaluation_at_eye_level()
    # test_cost_evaluation_at_waist_level()
    # test_cost_evaluation_at_arms_length()


def test_AUIT():
    """Test AUIT.py."""
    test_cost_evaluation()


def main():
    """Main function."""
    test_AUIT()


if __name__ == "__main__":
    main()
