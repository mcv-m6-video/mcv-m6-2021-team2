import numpy as np
from typing import List, Tuple, Dict

from src.metrics.ap import AP


def mAP(y_true, y_pred, classes=None, sort_method=None):
    if classes is None:
        classes = np.unique([det.label for boxlist in y_true for det in boxlist])

    precs = []
    recs = []
    aps = []
    for cls in classes:
        y_true_cls = [[det for det in boxlist if det.label == cls] for boxlist in y_true]
        y_pred_cls = [[det for det in boxlist if det.label == cls] for boxlist in y_pred]
        ap, prec, rec = AP(y_true_cls, y_pred_cls, sort_method)
        precs.append(prec)
        recs.append(rec)
        aps.append(ap)
    prec = np.mean(precs) if aps else 0
    rec = np.mean(recs) if aps else 0
    map = np.mean(aps) if aps else 0

    return map, prec, rec
