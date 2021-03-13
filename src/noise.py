from typing import List
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from src.annotation import Annotation

def apply_noise_to_bounding_boxes(annotations: List[Annotation], noise):
    if isinstance(bounding_boxes, list):
        noisy_bb = _apply_noise_to_list(bounding_boxes, noise)
    elif isinstance(bounding_boxes, OrderedDict):
        noisy_bb = _apply_noise_to_dict(bounding_boxes, noise)
    else:
        raise ValueError(f'The bounding boxes type is not supported: {type(bounding_boxes)}')
    return noisy_bb

def _apply_noise_to_list(bounding_boxes, noise, ):
    noisy_bounding_boxes = []
    for bbox in bounding_boxes:
        bb = deepcopy(bbox)
        if np.random.random() > noise['drop_probability']:
            box_noisy = bb.bbox + np.random.normal(noise['mean'], noise['std'], 4)
            bb.xtl = box_noisy[0]
            bb.ytl = box_noisy[1]
            bb.xbr = box_noisy[2]
            bb.ybr = box_noisy[3]
            noisy_bounding_boxes.append(bb)
    return noisy_bounding_boxes

def _apply_noise_to_dict(bounding_boxes, noise):
    noisy_bounding_boxes = []
    for key, bboxes in bounding_boxes.items():
        noisy_bounding_boxes += _apply_noise_to_list(bboxes, noise)
    return noisy_bounding_boxes