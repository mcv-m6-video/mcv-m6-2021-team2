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
    score: float = None
    guid: str = None

    def __post_init__(self):
        self.guid = str(uuid4())

    def get_bbox(self):
        return [self.left, self.top, self.width, self.height]