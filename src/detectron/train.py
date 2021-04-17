from typing import OrderedDict, NoReturn, List

import torch
import cv2
import os
import shutil
from pathlib import Path

from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

def get_dicts(frames_idx: List[int],
              annotations: OrderedDict) -> NoReturn:

    dataset_dicts = []

    for frame_idx in frames_idx:
        if frame_idx in annotations:
            record = {}

            filename = annotations[frame_idx][0].frame_path
            height, width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = frame_idx
            record["height"] = height
            record["width"] = width

            objs = []
            for det in annotations[frame_idx]:
                obj = {
                    "bbox": det.bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
                objs.append(obj)
            """
            record["annotations"] = objs
            print(annotations[frame_idx])
            print(record)
            img = cv2.imread(filename)

            for det in objs:
                cv2.rectangle(img, (int(det['bbox'][0]), int(det['bbox'][1])), (int(det['bbox'][2]), int(det['bbox'][3])), (0, 255, 0), 2)
            cv2.imwrite(f'obj_frame_{frame_idx}.png', img)

            for det in annotations[frame_idx]:
                bbox = det.bbox
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(+bbox[3])), (0, 255, 0), 2)
            cv2.imwrite(f'gt_frame_{frame_idx}.png', img)
            exit(1)
            """

            dataset_dicts.append(record)

    return dataset_dicts


def train(model_name: str,
          results_path: str,
          train_idx: List[int],
          test_idx: List[int],
          train_annotations: OrderedDict,
          test_annotations: OrderedDict,
          lr: float = 0.0025,
          max_it: int = 500,
          img_per_batch: int = 16,
          batch_size: int = 512,
          num_freeze: int = 1) -> NoReturn:

    if Path(results_path).exists():
        shutil.rmtree(results_path)

    os.makedirs(results_path, exist_ok=True)

    for catalog_type in ['train', 'test']:
        catalog = f'aic19_{catalog_type}'
        if catalog in DatasetCatalog.list():
            DatasetCatalog.remove(catalog)

        if catalog_type == 'train':
            DatasetCatalog.register(catalog, lambda d=catalog_type: get_dicts(train_idx, train_annotations))
        else:
            DatasetCatalog.register(catalog, lambda d=catalog_type: get_dicts(test_idx, test_annotations))

        MetadataCatalog.get(catalog).set(thing_classes=['Car'])

    cfg = get_cfg()
    cfg.OUTPUT_DIR = results_path
    cfg.merge_from_file(model_zoo.get_config_file(model_name))

    cfg.DATASETS.TRAIN = (f'aic19_train',)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 16

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    #cfg.MODEL.BACKBONE.FREEZE_AT = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.INPUT.MAX_SIZE_TEST = 1200
    cfg.INPUT.MAX_SIZE_TRAIN = 1200

    cfg.SOLVER.IMS_PER_BATCH = img_per_batch
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_it
    cfg.SOLVER.STEPS = []

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

    evaluator = COCOEvaluator('aic19_test', cfg, False, output_dir=results_path)
    val_loader = build_detection_test_loader(cfg, "aic19_test")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
