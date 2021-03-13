from typing import List, Tuple
import numpy as np

from src.annotation import Annotation
from src.metrics.voc_ap import voc_ap

def AP(pred_annons: List[Annotation],
       gt_annons: List[Annotation],
       score_available: bool = False,
       N: int = 10,
       th: float = 0.5) -> Tuple[float, float, float]:

    if score_available:
        pred_list_sorted = sorted(pred_annons, key=lambda x: x.score, reverse=True)
        return voc_ap(pred_list_sorted, gt_annons, th)

    precisions = []
    recalls = []
    aps = []

    for _ in range(N):
        shuffled_ind = np.random.permutation(len(pred_list))
        pred_list_shuffled = [pred_list[i] for i in shuffled_ind]
        ap, prec, rec = voc_ap(pred_list_shuffled, gt_list)

        precs.append(prec)
        recs.append(rec)
        aps.append(ap)

    return np.mean(aps), np.mean(precisions), np.mean(recalls)