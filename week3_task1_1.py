# TODO
# * Compute the mAP using YOLO3
# * Compute the mAP using SSD512
# * Compute the mAP using Mask R-CNN
# * Compute the mAP using Faster R-CNN
# * Compute the mAP using Retina (optional)

import logging
from pathlib import Path

from src.models.pre_trained import torchvision_inference
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.metrics.map import mAP

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

video_path = str(Path.joinpath(Path(__file__).parent, './data/vdo.avi'))
gt_path = str(Path.joinpath(Path(__file__).parent, './data/s03_c010-annotation.xml'))
result_path = str(Path.joinpath(Path(__file__).parent, './results/week3/s03_c010-fast-rcnn.txt'))

"""
torchvision_inference(model_name='fasterrcnn',
                      video_path=video_path,
                      results_path=result_path,
                      labels=[3])
"""

pred_reader = AICityChallengeAnnotationReader(result_path)
pred_annotations = pred_reader.get_annotations()

gt_reader = AICityChallengeAnnotationReader(gt_path)
gt_annotations = gt_reader.get_annotations()

y_true = []
y_pred = []
for frame in pred_annotations.keys():

    y_true.append(gt_annotations.get(frame, []))
    y_pred.append(pred_annotations.get(frame))

ap, prec, rec = mAP(y_true, y_pred, classes=['car'])
print(f'AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')