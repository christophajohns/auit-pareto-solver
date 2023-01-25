"""Element class."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Element:
    """An element is a UI element including its position
    as a 3-element vector and its rotation as a 3-element vector."""

    id: str = "0"
    position: List[float] = field(
        default_factory=lambda: [0, 0, 0]
    )  # x, y, z (Vector3)
    rotation: List[float] = field(
        default_factory=lambda: [0, 0, 0, 1]
    )  # x, y, z, w (Quaternion)

    def __dict__(self):
        """Return a dictionary representation of the UI element."""
        return {
            "id": self.id,
            "position": self.position,
            "rotation": self.rotation,
        }
