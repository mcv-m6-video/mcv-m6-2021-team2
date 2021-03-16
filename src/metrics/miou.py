from typing import Dict, List
import numpy as np

from src.metrics.iou import IoU
from src.annotation import Annotation


def mIoU(pred_annons: List[Annotation],
         gt_annons: List[Annotation],
         score_available: bool = True) -> float:
    if score_available:
        pred_annons = sorted(pred_annons, key=lambda x: x.score, reverse=True)
    else:
        pred_annons = np.random.shuffle(pred_annons)

    gt_frames = {}
    for annon in gt_annons:
        frame = annon.frame

        if frame not in gt_frames:
            gt_frames[frame] = {'annon': [], 'used': []}
        gt_frames[frame]['annon'].append(annon)
        gt_frames[frame]['used'].append(False)

    IoUs = []
    for pred_annon in pred_annons:
        gt = gt_frames[pred_annon.frame]

        if gt:
            IoUs.append(np.max([IoU(x, pred_annon) for x in gt['annon']]))
        else:
            IoUs.append(0)

    return np.mean(IoUs)

def mIoU_by_frame(pred_annons: List[Annotation],
                  gt_annons: List[Annotation],
                  score_available: bool = True) -> Dict[int, float]:

    if score_available:
        pred_annons = sorted(pred_annons, key=lambda x: x.score, reverse=True)
    else:
        pred_annons = np.random.shuffle(pred_annons)

    gt_frames = {}
    for annon in gt_annons:
        frame = annon.frame

        if frame not in gt_frames:
            gt_frames[frame] = {'annon': [], 'used': []}
        gt_frames[frame]['annon'].append(annon)
        gt_frames[frame]['used'].append(False)

    frame_miou = {}
    for pred_annon in pred_annons:
        if pred_annon.frame not in frame_miou:
            frame_miou[pred_annon.frame] = []

        gt = gt_frames[pred_annon.frame]
        iou = 0

        if gt:
            iou = np.max([IoU(x, pred_annon) for x in gt['annon']])

        frame_miou[pred_annon.frame].append(iou)

    for k, miou in frame_miou.items():
        frame_miou[k] = np.mean(miou)

    return frame_miou