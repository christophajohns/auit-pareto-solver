import AUIT
from typing import Literal, Callable, Optional

OBJECTIVE = Literal["neck", "shoulder", "torso", "reach"]

EYE_POSITION = AUIT.networking.element.Position(x=0.0, y=0.0, z=0.0)
SHOULDER_JOINT_POSITION = AUIT.networking.element.Position(x=0.0, y=-0.3, z=0.0)
ARM_LENGTH = 0.7
WAIST_POSITION = AUIT.networking.element.Position(x=0.0, y=-0.7, z=0.0)

OBJECTIVE_FUNCTIONS = {
    "neck": lambda adaptation: sum([AUIT.get_neck_ergonomics_cost(eye_position=EYE_POSITION, element=item) for item in adaptation.items]),
    "shoulder": lambda adaptation: sum([AUIT.get_arm_ergonomics_cost(shoulder_joint_position=SHOULDER_JOINT_POSITION, element=item) for item in adaptation.items]),
    "torso": lambda adaptation: sum([AUIT.get_torso_ergonomics_cost(waist_position=WAIST_POSITION, element=item) for item in adaptation.items]),
    "reach": lambda adaptation: sum([AUIT.get_at_arms_length_cost(shoulder_joint_position=SHOULDER_JOINT_POSITION, arm_length=ARM_LENGTH, element=item) for item in adaptation.items]),
}

UTILITY_FUNCTION = Callable[[AUIT.networking.layout.Layout, Optional[bool]], float]