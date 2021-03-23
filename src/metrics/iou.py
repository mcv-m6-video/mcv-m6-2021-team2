import numpy as np


def IoU(boxes1, boxes2):

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    # intersection
    ixmin = np.maximum(x11, np.transpose(x21))
    iymin = np.maximum(y11, np.transpose(y21))
    ixmax = np.minimum(x12, np.transpose(x22))
    iymax = np.minimum(y12, np.transpose(y22))
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    area1 = (x12 - x11 + 1.0) * (y12 - y11 + 1.0)
    area2 = (x22 - x21 + 1.0) * (y22 - y21 + 1.0)
    uni = area1 + np.transpose(area2) - inters

    overlaps = inters / uni

    return overlaps


def iou_single_boxes(box1, box2):

    # determine intersection rectangle coordinates
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])

    # check non-overlapping rectangle
    if right < left or bottom < top:
        return 0

    # compute intersection area
    intersection_area = (right - left) * (bottom - top)

    # compute area of bbs
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # compute IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou
