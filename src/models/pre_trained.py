import logging
import torch
import torchvision
import numpy as np

from pathlib import Path
from typing import NoReturn, List

from torchvision.transforms import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.video import get_frames_from_video


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
                        result_file.write(f'{frame_idx},{label.item()},{box[0]},{box[1]},{box[2]},{box[3]},{score.item()},-1,-1,-1\n')
