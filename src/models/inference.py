import logging
import torch
import torchvision
import numpy as np

from pathlib import Path
from typing import NoReturn, List, Tuple

from torchvision.transforms import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.video import get_frames_from_video

def filter_by_label(pred_boxes: np.array,
                    pred_labels: np.array,
                    pred_scores: np.array,
                    labels: List[int]) -> Tuple[np.array, np.array, np.array]:

    for label in labels:
        filtered_idx = np.argwhere(pred_labels == label).flatten()
        pred_boxes = pred_boxes[filtered_idx]
        pred_labels = pred_labels[filtered_idx]
        pred_scores = pred_scores[filtered_idx]

    return pred_boxes, pred_labels, pred_scores

def remove_overlap(pred_boxes: np.array,
                   pred_labels: np.array,
                   pred_scores: np.array):

    xtl = pred_boxes[:,0]
    ytl = pred_boxes[:,1]
    xbr = pred_boxes[:,2]
    ybr = pred_boxes[:,3]

    area = (xbr - xtl + 1) * (ybr - ytl + 1)
    idxs = np.argsort(ybr)

    pick = []
    while len(idxs) > 0:
            i = idxs[-1]

            pick.append(i)

            xx1 = np.maximum(xtl[i], xtl[idxs[:-1]])
            yy1 = np.maximum(ytl[i], ytl[idxs[:-1]])
            xx2 = np.minimum(xbr[i], xbr[idxs[:-1]])
            yy2 = np.minimum(ybr[i], ybr[idxs[:-1]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:-1]]

            idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1],
                np.where(overlap > 0.70)[0])))

    return pred_boxes[pick], pred_labels[pick], pred_scores[pick]

def torchvision_inference(model_name: str,
                          video_path: str,
                          results_path: str,
                          labels: List[int],
                          start_frame: int = 0,
                          end_frame: int = np.inf,
                          colorspace: str = 'rgb') -> NoReturn:

    if not torch.cuda.is_available():
        raise EnvironmentError(f'Error, no GPU detected.')

    device = torch.device('cuda')

    if model_name == 'fasterrcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == 'retinanet':
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    else:
        raise NotImplementedError(f'The model with name: {model_name} is not yet implemented.')

    logging.debug(f'Running inference using: {model_name} model ...')

    model.to(device)
    model.eval()

    with open(results_path, 'w') as result_file:
        tensor = transforms.ToTensor()
        with torch.no_grad():
            for frame_idx, frame in get_frames_from_video(video_path, colorspace, start_frame, end_frame):
                if frame is not None:
                    preds = model([tensor(frame).to(device)])[0]

                    pred_boxes = np.array([box.cpu().numpy() for box in preds['boxes']])
                    pred_labels = np.array([label.item() for label in preds['labels']])
                    pred_scores = np.array([score.item() for score in preds['scores']])

                    pred_boxes, pred_labels, pred_scores = filter_by_label(pred_boxes, pred_labels, pred_scores, labels)
                    pred_boxes, pred_labels, pred_scores = remove_overlap(pred_boxes, pred_labels, pred_scores)

                    for box, score in zip(pred_boxes, pred_scores):
                        xtl = box[0]
                        ytl = box[1]
                        xbr = box[2]
                        ybr = box[3]

                        result_file.write(f'{frame_idx},-1,{xtl},{ytl},{xbr - xtl},{ybr - ytl},{score},-1,-1,-1\n')
