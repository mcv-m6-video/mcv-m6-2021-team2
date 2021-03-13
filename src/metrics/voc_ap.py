from typing import Tuple, Dict
from functools import reduce
import numpy as np

from src.annotation import Annotation
from src.metrics.iou import IoU

def voc_ap(pred_annons: Dict[int, Annotation],
           gt_annons: Dict[int, Annotation],
           th: float = 0.5) -> Tuple[float, float, float]:

    gt_frames = {}

    for frame, annons in gt_annons.items():
        gt_frames.setdefault(frame, [False] * len(annons))

        """
        if frame not in gt_frames:
            gt_frames[frame] = {'annon': [], 'used': []}
        gt_frames[frame]['annon'].append(annon)
        gt_frames[frame]['used'].append(False)
        """

    pred_annons = reduce(lambda x,y: x+y, list(pred_annons.values()))
    tp = np.zeros(len(pred_annons))
    fp = np.zeros(len(pred_annons))

    for predict_idx, pred_annon in enumerate(pred_annons):
        iou = -np.inf
        gt = gt_annons[pred_annon.frame]

        if gt:
            ious = [IoU(x, pred_annon) for x in gt]
            iou = np.max(ious)
            idx_gt_used = np.argmax(ious)

        if iou > th:
            if not gt_frames[pred_annon.frame][idx_gt_used]:
                tp[predict_idx] = 1.0
                gt_frames[pred_annon.frame][idx_gt_used] = True
            else:
                fp[predict_idx] = 1.0
        else:
            fp[predict_idx] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(len(gt_annons))

    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return ap, prec, rec