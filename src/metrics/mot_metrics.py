import motmetrics
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class IDF1Computation:

    def __init__(self):
        self.acc = motmetrics.MOTAccumulator(auto_id=True)

    def add_frame_detections(self, y_true, y_pred):
        X = [det.center for det in y_true]
        Y = [det.center for det in y_pred]
        dists = motmetrics.distances.norm2squared_matrix(X, Y)

        self.acc.update(
            [det.id for det in y_true],
            [det.id for det in y_pred],
            dists
        )

    def get_computation(self):
        mh = motmetrics.metrics.create()
        summary = mh.compute(self.acc, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='metrics')
        return summary
