"""Layout class."""

from dataclasses import dataclass
from typing import List
from .element import Element


@dataclass
class Layout:
    """A layout is a set of UI elements."""

    elements: List[Element]

    def __dict__(self):
        """Return a dictionary representation of the layout."""
        return {"elements": [element.__dict__() for element in self.elements]}

    def from_dict(data):
        """Return a layout from a dictionary."""
        return Layout(
            elements=[
                Element(
                    id=element["id"],
                    position=element["position"],
                    rotation=element["rotation"],
                )
                for element in data["elements"]
            ]
        )

    @property
    def n_elements(self):
        """Return the number of elements in the layout."""
        return len(self.elements)
