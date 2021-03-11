
class BoundingBox():
    def __init__(self, frame, instance_id, label, xtl, ytl, xbr, ybr, score):
        self.frame = frame
        self.instance_id = instance_id
        self.label = label
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.confidence = score

    @property
    def bbox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    @property
    def width(self):
        return abs(self.xbr - self.xtl)

    @property
    def height(self):
        return abs(self.ytl - self.ybr)

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (int((self.xtl + self.xbr) / 2), int((self.ybr + self.ytl) / 2))

    def __repr__(self):
        return f'BoundingBox:: frame:{self.frame}, instance_id:{self.instance_id}, label: {self.label}'