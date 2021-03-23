import logging
import cv2
import os
import numpy as np
from threading import Thread
from pathlib import Path

from src.models.train import torchvision_train, detectron_train
from src.models.inference import detectron_inference
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.video import generate_video
from src.metrics.map import mAP
from src.video import get_frames_from_video


video_path = str(Path.joinpath(Path(__file__).parent, './data/vdo.avi'))


def task1_2(generate_video_frames: bool = False):
    if generate_video_frames:
        os.makedirs(str(Path.joinpath(Path(__file__).parent, './frames')), exist_ok=True)

        for frame_idx, frame in get_frames_from_video(video_path):
            if frame is not None:
                cv2.imwrite(str(Path.joinpath(Path(__file__).parent, f'./frames/frame{frame_idx-1}.png')), frame)

    gt_path = str(Path.joinpath(Path(__file__).parent, './data/s03_c010-annotation.xml'))

    gt_reader = AICityChallengeAnnotationReader(gt_path)
    gt_annotations = gt_reader.get_annotations(classes=['car'])

    indices = list(gt_annotations.keys())
    test_idx = indices[:int(len(indices)*0.25)]
    train_idx = indices[int(len(indices)*0.25):]

    detectron_train(train_idx, test_idx, gt_annotations)

    detectron_inference("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
                        video_path,
                        "prueba.txt",
                        [0],
                        0,
                        len(test_idx),
                        'rgb',
                        str(Path.joinpath(Path(__file__).parent, './results/model_final.pth')))

    pred_reader = AICityChallengeAnnotationReader('prueba.txt')
    pred_annotations = pred_reader.get_annotations(classes=['car'])

    gt_path = str(Path.joinpath(Path(__file__).parent, './data/s03_c010-annotation.xml'))

    gt_reader = AICityChallengeAnnotationReader(gt_path)
    gt_annotations = gt_reader.get_annotations(classes=['car'])

    y_true = []
    y_pred = []
    for frame in pred_annotations.keys():

        y_true.append(gt_annotations.get(frame, []))
        y_pred.append(pred_annotations.get(frame))

    ap, prec, rec = mAP(y_true, y_pred, classes=['car'])
    print(f'Arch: Faster R-CNN, AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

