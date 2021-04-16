import random


class Track(object):

    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.tracking = detection if isinstance(detection, list) else [detection]
        self.color = (int(random.random() * 256),
                      int(random.random() * 256),
                      int(random.random() * 256))
        self.finished = False

    def add_detection_to_tracking(self, detection):
        self.tracking.append(detection)

    @property
    def previous_detection(self):
        return self.tracking[-1]
