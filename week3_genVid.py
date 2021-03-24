import logging
import cv2
import numpy as np
from threading import Thread
from pathlib import Path

from src.models.inference import torchvision_inference, detectron_inference
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.video import generate_video
from src.metrics.map import mAP

def genVid( use_txt: str = './results/week3/s03_c010-fasterrcnn_r_50_fpn_3x.txt',
            video_name: str = 'outputfasterrrrr.gif',
            video_title: str = 'Faster R-CNN'):

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
    video_path = str(Path.joinpath(Path(__file__).parent, './data/vdo.avi'))


    pred_reader = AICityChallengeAnnotationReader(use_txt)
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
    print(f'Arch: {video_title}, AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

    generate_video(video_path, video_name, pred_annotations, gt_annotations, video_title, 700, 900,)

if __name__ == "__main__":
    use_txt = 'faster_rcnn_R_50_FPN_3x_A.txt'
    video_name = "faster.gif"
    video_title = "Faster R-CNN"

    genVid(use_txt, video_name, video_title)
