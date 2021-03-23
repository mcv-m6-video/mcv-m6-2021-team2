import logging
import cv2
import numpy as np
from threading import Thread
from pathlib import Path

from src.models.inference import torchvision_inference, detectron_inference
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.video import generate_video
from src.metrics.map import mAP

def task1_1(detectron: bool = True,
            model_name: str = 'faster_rcnn_R_50_FPN_3x',
            result_path: str = './results/week3/s03_c010-fasterrcnn_r_50_fpn_3x.txt',
            create_video: bool = False,
            video_name: str = 'output.gif',
            video_title: str = 'dummy'):

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
    video_path = str(Path.joinpath(Path(__file__).parent, './data/vdo.avi'))
    result_path = str(Path.joinpath(Path(__file__).parent, result_path))

    if detectron:
        # Detectron inference
        detectron_inference(model_name=model_name,
                            video_path=video_path,
                            results_path=result_path,
                            labels=[2])
    else:
        # Torchvision inference
        torchvision_inference(model_name=model_name,
                            video_path=video_path,
                            results_path=result_path,
                            labels=[3])

    pred_reader = AICityChallengeAnnotationReader(result_path)
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

    if create_video:
        generate_video(video_path, video_name, pred_annotations, gt_annotations, video_title, 500, 800,)

if __name__ == "__main__":
    detectron = True
    model_name = 'faster_rcnn_R_50_FPN_3x'
    result_path = './results/week3/s03_c010-fasterrcnn_r_50_fpn_3x.txt'

    task1_1(detectron, model_name, result_path, True)