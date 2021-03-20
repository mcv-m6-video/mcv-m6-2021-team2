from typing import List, Tuple, Dict
import numpy as np

from src.metrics.voc_ap import voc_ap

def AP(y_true, y_pred, sort_method=None):
    y_pred = [(i, det) for i in range(len(y_pred)) for det in y_pred[i]]  # flatten
    if len(y_pred) == 0:
        return 0

    if sort_method == 'score':
        # sort by confidence
        sorted_ind = np.argsort([-det[1].score for det in y_pred])
        y_pred_sorted = [y_pred[i] for i in sorted_ind]
        ap, prec, rec = voc_ap(y_true, y_pred_sorted)
    elif sort_method == 'area':
        # sort by area
        sorted_ind = np.argsort([-det[1].area for det in y_pred])
        y_pred_sorted = [y_pred[i] for i in sorted_ind]
        ap, prec, rec = voc_ap(y_true, y_pred_sorted)
    else:
        # average metrics across n random orderings
        n = 10
        precs = []
        recs = []
        aps = []
        for _ in range(n):
            shuffled_ind = np.random.permutation(len(y_pred))
            y_pred_shuffled = [y_pred[i] for i in shuffled_ind]
            ap, prec, rec = voc_ap(y_true, y_pred_shuffled)
            precs.append(prec)
            recs.append(rec)
            aps.append(ap)
        prec = np.mean(precs)
        rec = np.mean(recs)
        ap = np.mean(aps)
    return ap, prec, rec