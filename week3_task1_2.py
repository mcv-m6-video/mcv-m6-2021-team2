import logging
import cv2
import numpy as np
from threading import Thread
from pathlib import Path

from src.models.train import torchvision_train, detectron_train
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.video import generate_video
from src.metrics.map import mAP
from src.video import get_frames_from_video


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

video_path = str(Path.joinpath(Path(__file__).parent, './data/vdo.avi'))

# Generate frames
"""
for frame_idx, frame in get_frames_from_video(video_path):
    if frame is not None:
        cv2.imwrite(str(Path.joinpath(Path(__file__).parent, f'./frames/frame{frame_idx}.png')), frame)
"""

gt_path = str(Path.joinpath(Path(__file__).parent, './data/s03_c010-annotation.xml'))

gt_reader = AICityChallengeAnnotationReader(gt_path)
gt_annotations = gt_reader.get_annotations(classes=['car'])

indices = list(gt_annotations.keys())
test_idx = indices[:int(len(indices)*0.25)]
train_idx = indices[int(len(indices)*0.25):]

detectron_train(train_idx, test_idx, gt_annotations)

"""


torchvision_train('fasterrcnn', video_path, 2, 10, gt_annotations)
"""