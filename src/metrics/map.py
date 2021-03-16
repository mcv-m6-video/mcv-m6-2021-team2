import numpy as np
from typing import List, Tuple, Dict

from src.annotation import Annotation
from src.metrics.ap import AP

def mAP(pred_annons: List[Annotation],
        gt_annons: List[Annotation],
        classes: List[str],
        score_available: bool = False,
        N: int = 10,
        th: float = 0.5) -> Tuple[float, float, float, Dict[str, Dict[int, float]]]:

    precisions = []
    recalls = []
    aps = []
    mious_per_class = {}

    for k in classes:
        class_k_pred_annons = [annon for annon in pred_annons if annon.label == k]
        class_k_gt_annons = [annon for annon in gt_annons if annon.label == k]

        ap, precision, recall, mious_per_frame = AP(class_k_pred_annons, class_k_gt_annons, score_available, N, th)

        precisions.append(precision)
        recalls.append(recall)
        aps.append(ap)
        mious_per_class[k] = mious_per_frame

    return np.mean(aps), np.mean(precisions), np.mean(recalls), mious_per_class