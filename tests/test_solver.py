"""Test functions for the Python Pareto solver"""

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


from networking.messages import OptimizationResponse
from tests.test_evaluation import get_layout_with_element_at_eye_level

def test_optimization_response():
    """Test the OptimizationResponse class."""
    # Test whether the OptimizationResponse class has a value for the "default" attribute
    # Construct a list of Layout objects
    layout = get_layout_with_element_at_eye_level()
    solutions = [layout]
    optimization_response = OptimizationResponse(solutions=solutions)
    assert optimization_response.solutions == solutions
    assert optimization_response.default == layout


def test_solver():
    """Test the Pareto solver."""
    test_optimization_response()


def main():
    """Main function."""
    test_solver()


if __name__ == "__main__":
    main()
