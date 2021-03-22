import motmetrics
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class IDF1Computation:

    def __init__(self):
        self.acc = motmetrics.MOTAccumulator(auto_id=True)

    def add_frame_detections(self, y_true, y_pred):
        X = np.array([det.center for det in y_true])
        Y = np.array([det.center for det in y_pred])

        if len(X) > 0 and len(Y) > 0:
            dists = pairwise_distances(X, Y, metric='euclidean')
        else:
            dists = np.array([])

        self.acc.update(
            [det.id for det in y_true],
            [det.id for det in y_pred],
            dists
        )

    def get_computation(self):
        mh = motmetrics.metrics.create()
        summary = mh.compute(self.acc, metrics=['idf1'], name='acc')
        return summary['idf1']['acc']
