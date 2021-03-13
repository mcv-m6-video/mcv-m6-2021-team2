from dataclasses import dataclass
from uuid import uuid4

@dataclass
class Annotation:
    frame: int
    left: float
    top: float
    width: float
    height: float
    label: str
    score: float

    def get_bbox(self):
        return [self.left, self.top, self.width, self.height]