import logging
import cv2
import os
import numpy as np
from threading import Thread
from pathlib import Path
from sklearn.model_selection import KFold

from src.models.train import torchvision_train, detectron_train
from src.models.inference import detectron_inference
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.video import generate_video
from src.metrics.map import mAP
from src.video import get_frames_from_video


def task1_2(generate_video_frames: bool = False,
            model_name: str = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yml',
            model_output_path: str = str(Path.joinpath(Path(__file__).parent, './detectron_models/')),
            strategy: str = 'A'):

    os.makedirs(model_output_path, exist_ok=True)
    video_path = str(Path.joinpath(Path(__file__).parent, './data/vdo.avi'))

    if generate_video_frames:
        os.makedirs(str(Path.joinpath(Path(__file__).parent, './frames')), exist_ok=True)

        for frame_idx, frame in get_frames_from_video(video_path):
            if frame is not None:
                cv2.imwrite(str(Path.joinpath(Path(__file__).parent, f'./frames/frame{frame_idx-1}.png')), frame)

    gt_path = str(Path.joinpath(Path(__file__).parent, './data/s03_c010-annotation.xml'))

    gt_reader = AICityChallengeAnnotationReader(gt_path)
    gt_annotations = gt_reader.get_annotations(classes=['car'])

    indices = list(gt_annotations.keys())

    if strategy == 'A':
        result_path = str(Path.joinpath(Path(__file__).parent, f'./results/week3/{Path(model_name).stem}_A.txt'))
        train_idx = indices[:int(len(indices)*0.25)]
        test_idx = indices[int(len(indices)*0.25):]
        detectron_train(train_idx=train_idx,
                        test_idx=test_idx,
                        annotations=gt_annotations,
                        model_output_path=model_output_path,
                        model_name=model_name,
                        results_path=result_path)
    if strategy == 'B':
        result_path = str(Path.joinpath(Path(__file__).parent, f'./results/week3/{Path(model_name).stem}_B.txt'))
        kf = KFold(n_splits=4)
        for train_idx, test_idx in kf.split(indices):
            detectron_train(train_idx=train_idx,
                            test_idx=test_idx,
                            annotations=gt_annotations,
                            model_output_path=model_output_path,
                            model_name=model_name,
                            results_path=result_path)
    if strategy == 'C':
        result_path = str(Path.joinpath(Path(__file__).parent, f'./results/week3/{Path(model_name).stem}_C.txt'))
        kf = KFold(n_splits=4, shuffle=True)
        for train_idx, test_idx in kf.split(indices):
            detectron_train(train_idx=train_idx,
                            test_idx=test_idx,
                            annotations=gt_annotations,
                            model_output_path=model_output_path,
                            model_name=model_name,
                            results_path=result_path)

    pred_reader = AICityChallengeAnnotationReader(result_path)
    pred_annotations = pred_reader.get_annotations(classes=['car'])

    y_true = []
    y_pred = []
    for frame in pred_annotations.keys():

        y_true.append(gt_annotations.get(frame, []))
        y_pred.append(pred_annotations.get(frame))

    ap, prec, rec = mAP(y_true, y_pred, classes=['car'])

    print(f'Strategy: {strategy} Arch: {model_name} AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

if __name__ == "__main__":
    model_name = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'

    for strategy in ['A', 'B', 'C']:
        task1_2(generate_video_frames=False,
                model_name=model_name,
                strategy=strategy)