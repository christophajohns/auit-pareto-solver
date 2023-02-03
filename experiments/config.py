import AUIT
from typing import Literal

OBJECTIVE = Literal["neck", "shoulder", "torso", "reach"]

OBJECTIVE_FUNCTIONS = {
    "neck": lambda adaptation: sum([AUIT.get_neck_ergonomics_cost(eye_position=EYE_POSITION, element=item) for item in adaptation.items]),
    "shoulder": lambda adaptation: sum([AUIT.get_arm_ergonomics_cost(shoulder_joint_position=SHOULDER_JOINT_POSITION, element=item) for item in adaptation.items]),
    "torso": lambda adaptation: sum([AUIT.get_torso_ergonomics_cost(waist_position=waist_position, element=item) for item in adaptation.items]),
    "reach": lambda adaptation: sum([AUIT.get_arm_ergonomics_cost(shoulder_joint_position=SHOULDER_JOINT_POSITION, element=item) for item in adaptation.items]),
}

EYE_POSITION = AUIT.networking.element.Position(x=0.0, y=0.0, z=0.0)
SHOULDER_JOINT_POSITION = AUIT.networking.element.Position(x=0.0, y=-0.3, z=0.0)
waist_position = AUIT.networking.element.Position(x=0.0, y=-0.7, z=0.0)