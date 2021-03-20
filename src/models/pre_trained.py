import logging
import torch
import torchvision
import numpy as np

from pathlib import Path
from typing import NoReturn, List

from torchvision.transforms import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.video import get_frames_from_video

def get_nms(detections, overlapThresh):
    boxes =[box[1].cpu().numpy() for box in detections]
    picked_boxes = non_max_suppression_fast(np.array(boxes), overlapThresh )
    detections_int = [ (det[0].cpu().item(), list(map(int, det[1].cpu().numpy())), det[2].cpu().item()) for det in detections]
    picked_detections = list(filter(lambda x: x[1] in picked_boxes, detections_int))

    return picked_detections

# Malisiewicz et al.
def non_max_suppression_fast(boxes: np.array, overlapThresh: float):

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    xtl = boxes[:,0]
    ytl = boxes[:,1]
    xbr = boxes[:,2]
    ybr = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (xbr - xtl + 1) * (ybr - ytl + 1)
    idxs = np.argsort(ybr)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(xtl[i], xtl[idxs[:last]])
        yy1 = np.maximum(ytl[i], ytl[idxs[:last]])
        xx2 = np.minimum(xbr[i], xbr[idxs[:last]])
        yy2 = np.minimum(ybr[i], ybr[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


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
    else:
        raise NotImplementedError(f'The model with name: {model_name} is not yet implemented.')

    model.to(device)
    model.eval()

    with open(results_path, 'w') as result_file:

        tensor = transforms.ToTensor()
        for frame_idx, frame in get_frames_from_video(video_path, colorspace, start_frame, end_frame):
            if frame is not None:
                preds = model([tensor(frame).to(device)])[0]

                pred_boxes = preds['boxes']
                pred_labels = preds['labels']
                pred_scores = preds['scores']

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if label.item() in labels:
                        box = box.tolist()
                        result_file.write(f'{frame_idx},-1,{box[0]},{box[1]},{box[2]},{box[3]},{score.item()},-1,-1,-1\n')
