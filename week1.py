from typing import List
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
import copy

from tools.annotation_reader import read_annotations
from src.metrics import IoU, mAP
from src.annotation import Annotation
from src.read_flow_img import read_flow_img
from src.flow_metrics import calc_optical_flow

def task11(dropout=None, generate_gt=None, noise=None):
    gt_path = Path.joinpath(Path(__file__).parent, "s03_c010-gt.txt")

    gt_annons = read_annotations(str(gt_path))
    predict_annons = read_annotations(str(gt_path))

    # Without any change
    mapp, miou = mAP(gt_annons, predict_annons)
    print(f"Without any change: mAP {mapp} - mIOU {miou}")

    # Random Dropout
    if dropout is not None:
        rgt_annons = copy.deepcopy(gt_annons)
        np.random.shuffle(rgt_annons)
        rgt_annons = rgt_annons[int(len(rgt_annons)*dropout):-1]

        mapp, miou = mAP(rgt_annons, predict_annons)
        print(f"Dropout of {dropout}: mAP {mapp} - mIOU {miou}")

    # Generate Gt
    """
    if generate_gt > 0:
        rgt_annons = copy.deepcopy(gt_annons)

        mapp, miou = mAP(rgt_annons, predict_annons)
        print(f"Generate of {generate_gt}: mAP {mapp} - mIOU {miou}")
    """

    # Apply std
    if noise is not None:
        rgt_annons = copy.deepcopy(gt_annons)

        for rgt_annon in rgt_annons:
            rgt_annon.left += np.random.normal(noise[0], noise[1])
            rgt_annon.top += np.random.normal(noise[0], noise[1])
            rgt_annon.width += np.random.normal(noise[0], noise[1])
            rgt_annon.height += np.random.normal(noise[0], noise[1])

        mapp, miou = mAP(rgt_annons, predict_annons)
        print(f"Dropout of {dropout}: mAP {mapp} - mIOU {miou}")

def task12():
    predict_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-annotation.xml")))

    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-mask_rcnn.txt")))
    mapp, _ = mAP(gt_annons, predict_annons)
    print(f"Mask rcnn mAP: {mapp}")

    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-ssd512.txt")))
    mapp, _ = mAP(gt_annons, predict_annons)
    print(f"SSD 512 mAP: {mapp}")

    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-yolo3.txt")))
    mapp, _ = mAP(gt_annons, predict_annons)
    print(f"Yolo 3 mAP: {mapp}")

def task13():
    pred_000045_10 = read_flow_img(str(Path.joinpath(Path(__file__).parent, "pred_000045_10.png")))
    pred_000157_10 = read_flow_img(str(Path.joinpath(Path(__file__).parent, "pred_000157_10.png")))
    gt_000045_10 = read_flow_img(str(Path.joinpath(Path(__file__).parent, "gt_000045_10.png")))
    gt_000157_10 = read_flow_img(str(Path.joinpath(Path(__file__).parent, "gt_000157_10.png")))

    print(calc_optical_flow(gt_000045_10, pred_000045_10))

for _ in range(3):
    task12()