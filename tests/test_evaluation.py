"""Test functions for AUIT.py"""

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

import experiments.user
import experiments.random_solver
import experiments.problem
import AUIT

# Hyperparameters
EYE_POSITION = AUIT.networking.element.Position(x=0.0, y=0.0, z=0.0)
SHOULDER_JOINT_POSITION = AUIT.networking.element.Position(x=0.0, y=-0.3, z=0.0)
ARM_LENGTH = 3.0

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

def test_scenario_1_utility():
    """Test utility function for scenario 1."""
    print("Testing utility for scenario 1...")

    # Get utility function
    get_utility = experiments.user.get_utility_function(
        objectives=["neck", "shoulder", "torso"],
        weights=1/3,
    )

    # Test utility at eye level (should be > 0)
    print("Testing utility at eye level...")
    element_at_eye_level = get_element_at_eye_level()
    utility_at_eye_level = get_utility(element_at_eye_level, verbose=True)
    assert (
        utility_at_eye_level > 0
    ), "Utility should be greater than 0. Got: {}".format(
        utility_at_eye_level
    )
    print("Utility at eye level: {}".format(utility_at_eye_level))
    print()

    # Test utility at waist level
    print("Testing utility at waist level...")
    element_at_waist_level = get_element_at_waist_level()
    element_at_waist_level.position.x += 0.01 # Move the UI element slightly away from the waist position
    utility_at_waist_level = get_utility(element_at_waist_level, verbose=True)
    assert (
        utility_at_waist_level > 0
    ), "Utility should be greater than 0. Got: {}".format(
        utility_at_waist_level
    )
    print("Utility at waist level: {}".format(utility_at_waist_level))
    print()

def test_scenario_2_to_4_utility():
    """Test utility function for scenario 2 to 4."""
    print("Testing utility for scenario 2 to 4...")

    # Get utility function
    get_utility = experiments.user.get_utility_function(
        objectives=["neck", "shoulder"],
        weights=1/2,
    )

    # Test utility at eye level (should be > 0)
    print("Testing utility at eye level...")
    element_at_eye_level = get_element_at_eye_level()
    utility_at_eye_level = get_utility(element_at_eye_level, verbose=True)
    assert (
        utility_at_eye_level > 0
    ), "Utility should be greater than 0. Got: {}".format(
        utility_at_eye_level
    )
    print("Utility at eye level: {}".format(utility_at_eye_level))
    print()

    # Test utility at waist level
    print("Testing utility at waist level...")
    element_at_waist_level = get_element_at_waist_level()
    utility_at_waist_level = get_utility(element_at_waist_level, verbose=True)
    assert (
        utility_at_waist_level > 0
    ), "Utility should be greater than 0. Got: {}".format(
        utility_at_waist_level
    )
    print("Utility at waist level: {}".format(utility_at_waist_level))
    print()


def test_utility_functions():
    """Test various utility functions for the user models."""
    test_scenario_1_utility()
    test_scenario_2_to_4_utility()

def test_random_solver():
    """Test random solver."""
    print("Testing random solver...")

    # Get problem
    problem = experiments.problem.LayoutProblem(objectives=["neck", "shoulder"], weights=1/2)

    # Test random solver with 1 adaptation
    single_random_adaptation_layout = experiments.random_solver.get_random_adaptations(problem=problem, n_adaptations=1)

    assert (
        isinstance(single_random_adaptation_layout, AUIT.networking.layout.Layout)
    ), "Adaptation should be a layout. Got: {}".format(
        type(single_random_adaptation_layout)
    )
    assert (
        problem.layout_is_valid(layout=single_random_adaptation_layout)
    ), "Adaptation should be valid. Got: {}".format(
        single_random_adaptation_layout
    )
    print("Random adaptation layout: {}".format(single_random_adaptation_layout))
    print()

    # Test random solver with 10 adaptations
    multiple_random_adaptation_layouts = experiments.random_solver.get_random_adaptations(problem=problem, n_adaptations=10)

    assert (
        isinstance(multiple_random_adaptation_layouts, list)
    ), "Adaptations should be a list. Got: {}".format(
        type(multiple_random_adaptation_layouts)
    )

    for adaptation in multiple_random_adaptation_layouts:
        assert (
            isinstance(adaptation, AUIT.networking.layout.Layout)
        ), "Adaptation should be a layout. Got: {}".format(
            type(adaptation)
        )
        assert (
            problem.layout_is_valid(layout=adaptation)
        ), "Adaptation should be valid. Got: {}".format(
            single_random_adaptation_layout
        )

    print("Random adaptation layouts: {} layouts".format(len(multiple_random_adaptation_layouts)))
    print()




def test_evaluation():
    """Test evaluations."""
    test_utility_functions()
    test_random_solver()


def main():
    """Main function."""
    test_evaluation()


if __name__ == "__main__":
    main()
