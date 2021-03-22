# TODO
# * Compute the mAP using YOLO3
# * Compute the mAP using SSD512
# * Compute the mAP using Mask R-CNN
# * Compute the mAP using Faster R-CNN
# * Compute the mAP using Retina (optional)

import logging
import cv2
import numpy as np
from threading import Thread
from pathlib import Path

from src.models.inference import torchvision_inference
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.video import generate_video
from src.metrics.map import mAP


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

video_path = str(Path.joinpath(Path(__file__).parent, './data/vdo.avi'))
gt_path = str(Path.joinpath(Path(__file__).parent, './data/s03_c010-annotation.xml'))

retinanet_result_path = str(Path.joinpath(Path(__file__).parent, './results/week3/s03_c010-retinanet.txt'))
faster_rcnn_result_path = str(Path.joinpath(Path(__file__).parent, './results/week3/s03_c010-faster_rcnn.txt'))


torchvision_inference(model_name='fasterrcnn',
                      video_path=video_path,
                      results_path=faster_rcnn_result_path,
                      labels=[3])

torchvision_inference(model_name='retinanet',
                      video_path=video_path,
                      results_path=retinanet_result_path,
                      labels=[3])

pred_reader = AICityChallengeAnnotationReader(faster_rcnn_result_path)
pred_fasterrcnn_annotations = pred_reader.get_annotations(classes=['car'])

pred_reader = AICityChallengeAnnotationReader(retinanet_result_path)
pred_retinanet_annotations = pred_reader.get_annotations(classes=['car'])

gt_reader = AICityChallengeAnnotationReader(gt_path)
gt_annotations = gt_reader.get_annotations(classes=['car'])

# Faster R-CNN

y_true = []
y_pred = []
for frame in pred_fasterrcnn_annotations.keys():

    y_true.append(gt_annotations.get(frame, []))
    y_pred.append(pred_fasterrcnn_annotations.get(frame))

ap, prec, rec = mAP(y_true, y_pred, classes=['car'])
print(f'Arch: Faster R-CNN, AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

# Retina Net

y_true = []
y_pred = []
for frame in pred_retinanet_annotations.keys():

    y_true.append(gt_annotations.get(frame, []))
    y_pred.append(pred_retinanet_annotations.get(frame))

ap, prec, rec = mAP(y_true, y_pred, classes=['car'])
print(f'Arch: Retina net, AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')