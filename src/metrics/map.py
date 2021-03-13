import numpy as np
from typing import Dict, Tuple, List

from src.annotation import Annotation
from src.metrics.ap import AP

def mAP(pred_annons: Dict[int, Annotation],
        gt_annons: Dict[int, Annotation],
        classes: List[str],
        score_available: bool = False,
        N: int = 10,
        th: float = 0.5) -> Tuple[float, float, float]:

    precisions = []
    recalls = []
    aps = []

    for k in classes:
        class_k_pred_annons = {frame: [annon for annon in annons if annon.label == k] for frame, annons in pred_annons.items()}
        class_k_gt_annons = {frame: [annon for annon in annons if annon.label == k] for frame, annons in gt_annons.items()}

        ap, precision, recall = AP(class_k_pred_annons, class_k_gt_annons, score_available, N, th)

        precisions.append(precision)
        recalls.append(recall)
        aps.append(ap)

    return np.mean(aps), np.mean(precisions), np.mean(recalls)
