from typing import List, Tuple, Dict
import numpy as np

from src.annotation import Annotation
from src.metrics.voc_ap import voc_ap

def AP(pred_annons: List[Annotation],
       gt_annons: List[Annotation],
       score_available: bool = False,
       N: int = 10,
       th: float = 0.5) -> Tuple[float, float, float, Dict[int, float]]:

    if score_available:
        pred_list_sorted = sorted(pred_annons, key=lambda x: x.score, reverse=True)
        return voc_ap(pred_list_sorted, gt_annons, th)

    precisions = []
    recalls = []
    aps = []
    mious_list = []

    for _ in range(N):
        np.random.shuffle(pred_annons)
        ap, prec, rec, miou = voc_ap(pred_annons, gt_annons, th)

        precisions.append(prec)
        recalls.append(rec)
        aps.append(ap)
        mious_list.append(miou)

    mious_per_frame = {}

    for miou_per_frame in mious_list:
        for frame_idx, miou in miou_per_frame.items():
            mious_per_frame.setdefault(frame_idx, []).append(miou)

    mious_per_frame = {k: np.mean(v) for k, v in mious_per_frame.items()}

    return np.mean(aps), np.mean(precisions), np.mean(recalls), mious_per_frame