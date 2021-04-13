from dataclasses import dataclass


@dataclass
class Detection:
    frame: int
    id: int
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    video_seq: str = None
    cam: str = None
    score: float = None
    parked: bool = None

    @property
    def bbox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    @property
    def width(self):
        return abs(self.xbr - self.xtl)

    @property
    def height(self):
        return abs(self.ybr - self.ytl)

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (int((self.xtl + self.xbr) / 2), int((self.ytl + self.ybr) / 2))
