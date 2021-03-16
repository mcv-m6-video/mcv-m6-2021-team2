from typing import Tuple, List
import numpy as np

from src.annotation import Annotation
from src.metrics.iou import IoU

def voc_ap(pred_annons: List[Annotation],
           gt_annons: List[Annotation],
           th: float = 0.5) -> Tuple[float, float, float, List[Tuple[int, float]]]:

    gt_frames = {}
    for annon in gt_annons:
        frame = annon.frame

        if frame not in gt_frames:
            gt_frames[frame] = {'annon': [], 'used': []}
        gt_frames[frame]['annon'].append(annon)
        gt_frames[frame]['used'].append(False)

    tp = np.zeros(len(pred_annons))
    fp = np.zeros(len(pred_annons))
    mious = {}

    for predict_idx, pred_annon in enumerate(pred_annons):
        iou = 0
        gt = gt_frames[pred_annon.frame]

        if gt:
            ious = [IoU(x, pred_annon) for x in gt['annon']]
            iou = np.max(ious)
            idx_gt_used = np.argmax(ious)

        if iou > th:
            if not gt['used'][idx_gt_used]:
                tp[predict_idx] = 1.0
                gt['used'][idx_gt_used] = True
            else:
                fp[predict_idx] = 1.0
        else:
            fp[predict_idx] = 1.0

        mious.setdefault(predict_idx, []).append(iou)

    for predict_idx, miou in mious.items():
        mious[predict_idx] = np.mean(miou)

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

    return ap, prec, rec, mious