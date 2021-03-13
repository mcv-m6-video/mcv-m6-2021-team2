from typing import Dict, Tuple, List
import numpy as np

from src.annotation import Annotation
from src.metrics.voc_ap import voc_ap

def AP(pred_annons: Dict[int, Annotation],
       gt_annons: Dict[int, Annotation],
       score_available: bool = False,
       N: int = 10,
       th: float = 0.5) -> Tuple[float, float, float]:

    if score_available:
        return voc_ap(pred_annons, gt_annons, th)

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