from typing import OrderedDict

import numpy as np
import torch
import torchvision

from src.detection import Detection
from src.video import get_frame_from_video, get_video_lengh


class AICity(torch.utils.data.Dataset):
    def __init__(self,
                 video_path: str,
                 detections: OrderedDict):

        #self._transforms = transforms
        self._detections = detections
        self._video_path = video_path
        self._transforms = torchvision.transforms.ToTensor()

    def __getitem__(self, frame_idx: int):
        img = get_frame_from_video(self._video_path, frame_idx)

        img = self._transforms(img)

        boxes = torch.as_tensor([x.bbox for x in self._detections.get(frame_idx, [])], dtype=torch.float32)

        labels = torch.full((len(boxes),), 3, dtype=torch.int64)

        image_id = torch.tensor([frame_idx])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}

        return img, target

    def __len__(self):
        return get_video_lengh(self._video_path)