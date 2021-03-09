from typing import List
from src.annotation import Annotation

import numpy as np

def mAP(gt_annons: List[Annotation], predict_annons: List[Annotation], N=10):

    ap = []
    iou = []

    gt_frames = {}
    for gt_annon in gt_annons:
        frame = gt_annon.frame

        if frame not in gt_frames:
            gt_frames[frame] = []
        gt_frames[frame].append(gt_annon)

    for n in range(N):
        np.random.shuffle(predict_annons)

        gt_used = []

        tp = np.zeros(len(predict_annons))
        fp = np.zeros(len(predict_annons))

        for predict_idx, predict_annon in enumerate(predict_annons):
            IoUs = []
            gt_guids = []

            if predict_annon.frame in gt_frames:
                for gt_annon in gt_frames[predict_annon.frame]:
                    if gt_annon.guid not in gt_used:
                        IoUs.append(IoU(gt_annon, predict_annon))
                        gt_guids.append(gt_annon.guid)

            if IoUs:
                idx_max = np.asarray(IoUs, dtype=float).argmax(axis=0)

                if IoUs[idx_max] > 0.5:
                    tp[predict_idx] = 1
                    gt_used.append(gt_guids[idx_max])
                else:
                    fp[predict_idx] = 1

                iou.append(IoUs[idx_max])
            else:
                iou.append(0)
                fp[predict_idx] = 1

        ap.append(AP(tp, fp, len(gt_annons)))

    return np.mean(ap), np.mean(iou)

def AP(tp,fp,npos):
    """
    Calculate average precision from the given tp, fp and number of measurements
    Code modification from Detectron2: pascal_voc_evaluation.py
    (https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py)
    Input:
        tp: true positives bounding boxes
        fp: false positives bounding boxes
        npos: number of measurements
    Output:
        ap: average precision
    """

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # compute VOC AP using 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return ap

def IoU(gt_annon: Annotation, predict_annon: Annotation):
    """
    Src: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    gt_box = gt_annon.get_bbox()
    predict_box = predict_annon.get_bbox()

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(gt_box[0], predict_box[0])
    yA = max(gt_box[1], predict_box[1])
    xB = min(gt_box[2], predict_box[2])
    yB = min(gt_box[3], predict_box[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    boxBArea = (predict_box[2] - predict_box[0] + 1) * (predict_box[3] - predict_box[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou