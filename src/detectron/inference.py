from typing import OrderedDict, NoReturn, List

import torch
import cv2
import os
import shutil
import numpy as np
from pathlib import Path

from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from src.video import get_frames_from_video
from src.detectron.train import get_dicts

def inference_from_trained_model(video_path: str,
                                 model_name: str,
                                 results_path: str,
                                 val_idx: List[int],
                                 annotations: OrderedDict,
                                 frames_path: str,
                                 labels: List[int],
                                 weight_path: str,
                                 start_frame: int = 0,
                                 end_frame: int = np.inf,) -> NoReturn:

    DatasetCatalog.register('aic19_val', lambda d='val': get_dicts(val_idx, annotations, frames_path))
    MetadataCatalog.get('aic19_val').set(thing_classes=['Car'])

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = weight_path

    predictor = DefaultPredictor(cfg)

    with open(results_path, 'w') as result_file:
        for frame_idx, frame in get_frames_from_video(video_path, start_frame, end_frame):
            outputs = predictor(frame)



            print(outputs)
            torch.cuda.synchronize()

            pred_boxes = outputs["instances"].pred_boxes.to("cpu")
            pred_scores = outputs["instances"].scores.to("cpu")
            pred_labels = outputs["instances"].pred_classes.to("cpu")

            preds = zip(pred_boxes, pred_scores, pred_labels)

            preds = [(box.cpu().numpy(), score.item(), label.item()) for box, score, label in preds if label.item() in labels]

            for box, score, label in preds:
                xtl = box[0]
                ytl = box[1]
                xbr = box[2]
                ybr = box[3]

                result_file.write(f'{frame_idx},-1,{xtl},{ytl},{xbr - xtl},{ybr - ytl},{score},-1,-1,-1\n')

