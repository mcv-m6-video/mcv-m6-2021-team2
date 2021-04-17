from typing import List
from pathlib import Path
from src.readers.ai_city_reader import parse_annotations_from_txt, group_by_frame
from src.metrics.map import mAP

import cv2
import os
import numpy as np
import shutil

from src.detectron.train import train
from src.detectron.inference import inference_from_trained_model
from src.video import get_frames_from_video, generate_video, get_video_lenght

"""
video_seq_cam = {
    'S03' : ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'],
    'S01' : ['c001', 'c002', 'c003', 'c004', 'c005'],
    'S04' : [
        'c016', 'c017', 'c018', 'c019', 'c020', 'c021', 'c022', 'c023', 'c024', 'c025',
        'c026', 'c027', 'c028', 'c029', 'c030', 'c031', 'c032', 'c033', 'c034', 'c035',
        'c036', 'c037', 'c038', 'c039', 'c040'
    ]
}
"""

video_seq_cam = {
    'S03' : ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
}

def get_annotations_from_seq(seqs: List[str],
                             det: str):
    seq_annotations = {}
    offset = 0

    for seq in seqs:
        video_seq_path = str(Path.joinpath(Path(__file__).parent, f'./aic19-track1-mtmc-train/train/{seq}'))

        for cam in video_seq_cam[seq]:
            det_path = str(Path.joinpath(Path(video_seq_path), f'./{cam}/det/det_{det}.txt'))
            det_annotations = group_by_frame(parse_annotations_from_txt(det_path))

            for frame_idx, annotations in det_annotations.items():
                for annotation in annotations:
                    annotation.frame_path = str(Path.joinpath(Path(__file__).parent,
                                                f'./aic19_frames/{seq}/{cam}/frame_{frame_idx}.png').absolute())
                    annotation.id = offset + frame_idx
                seq_annotations[offset + frame_idx] = annotations

            offset = offset + get_video_lenght(str(Path.joinpath(Path(video_seq_path), f'./{cam}/vdo.avi')))

    return seq_annotations

def generate_frames_from_seqs(seqs: List[str]):
    for seq in seqs:
        video_seq_path = str(Path.joinpath(Path(__file__).parent, f'./aic19-track1-mtmc-train/train/{seq}'))

        for cam in video_seq_cam[seq]:
            frames_path = str(Path.joinpath(Path(__file__).parent, f'./aic19_frames/{seq}/{cam}'))
            video_path = str(Path.joinpath(Path(video_seq_path).parent, f'./{cam}/vdo.avi'))

            if Path(frames_path).exists():
                shutil.rmtree(frames_path)
            os.makedirs(frames_path)

            for frame_idx, frame in get_frames_from_video(video_path):
                cv2.imwrite(str(Path.joinpath(Path(frames_path).parent, f'frame_{frame_idx-1}.png')), frame)


def train_dataset(train_seqs: List[str],
                test_seqs: List[str],
                coco_architecture: str,
                model_name: str,
                lr: float,
                max_it: int,
                batch_size: int,
                img_per_batch: int,
                num_freeze: int,
                det: str,
                generate_frames: bool = False):

    train_annotations = get_annotations_from_seq(train_seqs, det)
    #test_annotations = get_annotations_from_seq(test_seqs, det)

    result_path = str(Path.joinpath(Path(__file__).parent, f'./results/week5/task1_1/detectron_{architecture}').absolute())

    list_idx = list(train_annotations.keys())
    np.random.shuffle(list_idx)

    test_idx = list_idx[int(len(list_idx)*0.25):]
    train_idx = list_idx[:int(len(list_idx)*0.25)]

    #test_idx = list(test_annotations.keys())
    #np.random.shuffle(test_idx)

    train(model_name=model_name,
          results_path=result_path,
          train_idx=train_idx,
          test_idx=test_idx,
          train_annotations=train_annotations,
          test_annotations=train_annotations,
          lr=lr,
          max_it=max_it,
          img_per_batch=img_per_batch,
          batch_size=batch_size,
          num_freeze=num_freeze)

if __name__ == "__main__":
    for architecture in ['faster_rcnn_R_50_FPN_3x']:
        train_dataset(train_seqs=['S03'],
                    test_seqs=[],
                    coco_architecture=architecture,
                    model_name=f'COCO-Detection/{architecture}.yaml',
                    lr=0.00001,
                    max_it=1000,
                    batch_size=512,
                    img_per_batch=8,
                    num_freeze=1,
                    det='mask_rcnn',
                    generate_frames=False
        )
