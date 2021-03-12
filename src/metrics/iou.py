import numpy as np

def voc_iou(gt, bb):
    """
    Compute IoU between groundtruth bounding box = gt
    and detected bounding box = bb
    """
    # intersection
    ixmin = np.maximum(gt[:, 0], bb[0])
    iymin = np.maximum(gt[:, 1], bb[1])
    ixmax = np.minimum(gt[:, 2], bb[2])
    iymax = np.minimum(gt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
          + (gt[:, 2] - gt[:, 0] + 1.0) * (gt[:, 3] - gt[:, 1] + 1.0)
          - inters)
    overlaps = inters/uni
    return overlaps

def mean_iou(det, gt, sort=False):
    '''
    det: detections of one frame
    gt: annotations of one frame
    sort: False if we use modified GT,
          True if we have a confidence value for the detection
    '''
    if sort:
        BB = sort_by_confidence(det)
    else:
        BB = np.array([x.bbox for x in det]).reshape(-1, 4)

    BBGT = np.array([anot.bbox for anot in gt])

    nd = len(BB)
    mean_iou = []
    for d in range(nd):
        bb = BB[d, :].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = voc_iou(BBGT, bb)
            ovmax = np.max(overlaps)
            mean_iou.append(ovmax)

    return np.mean(mean_iou)


def sort_by_confidence(det):
    BB = np.array([x.bbox for x in det]).reshape(-1, 4)
    confidence = np.array([float(x.confidence) for x in det])
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]

    return BB