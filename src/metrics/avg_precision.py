import numpy as np
from src.metrics.iou import voc_iou

"""
CODE ADAPTED FROM
https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py
"""


def voc_ap(rec, prec):
    """
    Compute VOC AP given precision and recall,
    using the 11-point method .
    """
    # 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0
    
    return ap


def voc_eval(predictions, targets, ovthresh=0.5, is_confidence=True):
    """
    rec, prec, ap = voc_eval(predictions,
                                targets
                                ovthresh)
    Top level function that does the PASCAL VOC -like evaluation.
    Predictions: the detected bounding boxes
    Targets: the groundtruth gropued by frames
    ovthresh: Overlap threshold (default = 0.5)
    """
    # read targets
    class_recs = {}
    npos = 0

    for frame_id, boxes in targets.items():
        bbox = np.array([bb.bbox for bb in boxes])
        det = [False] * len(boxes)
        npos += len(boxes)
        class_recs[frame_id] = {"bbox": bbox, "det": det}
 
    # read predictions
    image_ids = [x.frame for x in predictions]
    BB = np.array([x.bbox for x in predictions]).reshape(-1, 4)

    if is_confidence:
        confidence = np.array([float(x.confidence) for x in predictions])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

    # go down predictions (dets) and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = voc_iou(BBGT,bb)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap