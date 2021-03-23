from typing import OrderedDict, NoReturn, List

import torch
import logging
import os
import cv2
from pathlib import Path

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.models.torchvision_engine import train_one_epoch, evaluate
from src.models.torchvision_utils import collate_fn
from src.models.datasets.AICity import AICity
from src.models.TrainerDataAugmentation import WheatTrainer

from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode



def get_dicts(frames, annotations):

    dataset_dicts = []

    for frame_idx in frames:
        record = {}

        filename = str(Path.joinpath(Path(__file__).parent, f'../../frames/frame{frame_idx}.png'))
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

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def detectron_train(train_idx: List[int],
                    test_idx: List[int],
                    annotations: OrderedDict,
                    model_output_path: str,
                    model_name: str,
                    results_path: str):

    DatasetCatalog.register("aicity_train", lambda d='train': get_dicts(train_idx, annotations))
    MetadataCatalog.get("aicity_train").set(thing_classes=["car"])

    """
    DatasetCatalog.register("aicity_test", lambda d='test': get_dicts(test_idx, annotations))
    MetadataCatalog.get("aicity_test").set(thing_classes=["car"])
    """

    balloon_metadata = MetadataCatalog.get("aicity_train")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = model_output_path
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.DATASETS.TRAIN = ("aicity_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = WheatTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

    predictor = DefaultPredictor(cfg)

    with open(results_path, 'w') as result_file:
        for frame_idx in test_idx:
            filename = str(Path.joinpath(Path(__file__).parent, f'../../frames/frame{frame_idx}.png'))
            img = cv2.imread(filename)
            outputs = predictor(img)

            torch.cuda.synchronize()

            pred_boxes = outputs["instances"].pred_boxes.to("cpu")
            pred_scores = outputs["instances"].scores.to("cpu")
            pred_labels = outputs["instances"].pred_classes.to("cpu")

            preds = zip(pred_boxes, pred_scores, pred_labels)

            preds = [(box.cpu().numpy(), score.item(), label.item()) for box, score, label in preds if label.item() in [0]]

            for box, score, label in preds:
                xtl = box[0]
                ytl = box[1]
                xbr = box[2]
                ybr = box[3]

                result_file.write(f'{frame_idx},-1,{xtl},{ytl},{xbr - xtl},{ybr - ytl},{score},-1,-1,-1\n')


def torchvision_train(model_name: str,
                      video_path: str,
                      num_classes: int,
                      epochs: int,
                      detections: OrderedDict) -> NoReturn:

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    if not torch.cuda.is_available():
        raise EnvironmentError(f'Error, no GPU detected.')

    device = torch.device('cuda')
    dataset = AICity(video_path, detections)

    indices = range(len(dataset))
    split = int(len(dataset) * 0.25)

    train = torch.utils.data.SubsetRandomSampler(indices[:split])
    test = torch.utils.data.SubsetRandomSampler(indices[split:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, sampler=train, num_workers=1, collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset, batch_size=8, sampler=test, num_workers=1, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    logging.debug(f'Training using: {model_name} model ...')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    model.roi_heads.mask_roi_pool = None
    model.roi_heads.mask_head = None
    model.roi_heads.mask_predictor = None
    # move model to the right device
    model.to(device)

    params = [p for p in model.roi_heads.box_predictor.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)