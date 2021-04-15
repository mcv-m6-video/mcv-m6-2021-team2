from typing import List
from pathlib import Path
from src.readers.ai_city_reader import parse_annotations_from_txt
from src.metrics.map import mAP

import cv2
import os
import numpy as np

from src.detectron.train import train
from src.detectron.inference import inference_from_trained_model
from src.video import get_frames_from_video, generate_video, get_video_lenght

"""
def plot_sample():
    import json

    results_path = str(Path.joinpath(Path(__file__).parent, './results/week5/task1_1/aic19_S03/coco_instances_results.json'))

    with open(results_path, 'r') as f:
        results = json.load(f)

    frames = {}

    for instance in results[:5]:
        img_id = instance['image_id']
        bbox = instance['bbox']

        frame_path = str(Path.joinpath(Path(__file__).parent, f'./aic19_frames/S03_all_cams/frame_{img_id}.png'))
        img = cv2.imread(frame_path)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.imwrite(f'./tmp/frame_{img_id}.png', img)

def plot_gt():
    video_seq_path = str(Path.joinpath(Path(__file__).parent, f'./aic19-track1-mtmc-train/train/S03/c010/det/det_mask_rcnn.txt'))
    annotations = parse_annotations_from_txt(video_seq_path)

    i = 0
    for frame_idx, dets in annotations.items():
        frame_path = str(Path.joinpath(Path(__file__).parent, f'./aic19_frames/S03/c010/frame_{frame_idx}.png'))
        img = cv2.imread(frame_path)

        for det in dets:
            cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 2)
        cv2.imwrite(f'./tmp/frame_{frame_idx}.png', img)
        i = i + 1
        if i == 5:
            return

def gen_video():
    gt_path = str(Path.joinpath(Path(__file__).parent, f'./aic19-track1-mtmc-train/train/S03/c010/det/det_mask_rcnn.txt'))
    annotations = parse_annotations_from_txt(gt_path)
    video_path = str(Path.joinpath(Path(__file__).parent,
                   f'./aic19-track1-mtmc-train/train/S03/c011/vdo.avi'))
    generate_video(video_path, 'test.gif', annotations, annotations, 't', 0, 10)
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

def get_annotations_from_seq(seqs: List[str],
                             det: str):
    seq_annotations = {}
    offset = 0

    for seq in seqs:
        video_seq_path = str(Path.joinpath(Path(__file__).parent, f'./aic19-track1-mtmc-train/train/{seq}'))

        for cam in video_seq_cam[seq]:
            det_path = str(Path.joinpath(Path(video_seq_path), f'./{cam}/det/det_{det}.txt'))
            det_annotations = parse_annotations_from_txt(det_path)

            for frame_idx, annotations in det_annotations.items():
                for annotation in annotations:
                    annotation.frame_path = str(Path.joinpath(Path(__file__).absolute(),
                                                f'./aic19_frames/{seq}/{cam}/frame_{frame_idx-1}.png'))
                    annotation.id = offset + frame_idx
                seq_annotations[offset + frame_idx] = annotations

            offset = offset + get_video_lenght(str(Path.joinpath(Path(video_seq_path), f'./{cam}/vdo.avi')))

    return seq_annotations

def generate_frames_from_seqs(seqs: List[str]):
    for seq in seqs:
        video_seq_path = str(Path.joinpath(Path(__file__).parent, f'./aic19-track1-mtmc-train/train/{seq}'))

        for cam in video_seq_cam[seq]:
            frames_path = str(Path.joinpath(Path(__file__).parent, f'./aic19_frames/{seq}/{cam}'))
            video_path = str(Path.joinpath(Path(video_seq_path), f'./{cam}/vdo.avi'))

            for frame_idx, frame in get_frames_from_video(video_path):
                cv2.imwrite(str(Path.joinpath(Path(frames_path), f'frame_{frame_idx-1}.png')), frame)


def train_seq(train_seqs: List[str],
              test_seqs: List[str],
              model_name: str,
              lr: float,
              max_it: int,
              batch_size: int,
              img_per_batch: int,
              num_freeze: int,
              det: str,
              generate_frames: bool = True):

    train_seq_annotations = get_annotations_from_seq(train_seqs, det)
    test_seq_annotations = get_annotations_from_seq(test_seqs, det)

    if generate_frames:
        generate_frames_from_seqs(train_seqs + test_seqs)

    result_path = str(Path.joinpath(Path(__file__).parent, f'./results/week5/task1_1/detectron').absolute())

    train_idx = list(train_seq_annotations.keys())
    np.random.shuffle(train_idx)

    test_idx = list(test_seq_annotations.keys())
    np.random.shuffle(test_idx)

    train(model_name=model_name,
          results_path=result_path,
          train_idx=train_idx,
          test_idx=test_idx,
          annotations=seq_annotations,
          frames_path=frames_path,
          lr=lr,
          max_it=max_it,
          img_per_batch=img_per_batch,
          batch_size=batch_size,
          num_freeze=num_freeze)

"""
def inference(video_seq: str,
              cam: str,
              model_name: str):

    video_path = str(Path.joinpath(Path(__file__).parent,
                     f'./aic19-track1-mtmc-train/train/{video_seq}/{cam}/vdo.avi'))

    results_path = str(Path.joinpath(Path(__file__).parent,
                      f'./results/week5/task1_1/aic19_inference_{video_seq}_{cam}/predictions.txt').absolute())

    weight_path = str(Path.joinpath(Path(__file__).parent,
                      f'./results/week5/task1_1/aic19_S03/model_final.pth').absolute())

    os.makedirs(Path(results_path).parent, exist_ok=True)

    inference_from_trained_model(video_path=video_path,
                                 model_name=model_name,
                                 results_path=results_path,
                                 labels=[0],
                                 weight_path=weight_path)

    pred_annotations = parse_annotations_from_txt(results_path)

    gt_path = str(Path.joinpath(Path(__file__).parent,
                  f'./aic19-track1-mtmc-train/train/{video_seq}/{cam}/gt/gt.txt'))

    gt = parse_annotations_from_txt(gt_path)

    y_true = []
    y_pred = []
    for frame in pred_annotations.keys():

        y_true.append(gt_annotations.get(frame, []))
        y_pred.append(pred_annotations.get(frame))

    ap, prec, rec = mAP(y_true, y_pred, classes=['car'])
    print(f'AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

def train_single_cam(video_seq: str,
                     cam: str,
                     model_name: str,
                     lr: float,
                     max_it: int,
                     batch_size: int,
                     num_freeze: int):

    cam_path = str(Path.joinpath(Path(__file__).parent,
                   f'./aic19-track1-mtmc-train/train/{video_seq}/{cam}'))
    if not Path(cam_path).exists():
        raise ValueError(f'The cam path: {cam_path} does not exist.')

    video_path = str(Path.joinpath(Path(cam_path), './vdo.avi'))

    cam_gt_path = str(Path.joinpath(Path(cam_path), './det/det_mask_rcnn.txt'))

    annotations = parse_annotations_from_txt(cam_gt_path)

    frames_path = str(Path.joinpath(Path(__file__).parent, f'./aic19_frames/{video_seq}/{cam}'))
    if not Path(frames_path).exists():
        os.makedirs(frames_path)

        for frame_idx, frame in get_frames_from_video(video_path):
            if (frame_idx - 1) in annotations:
                cv2.imwrite(str(Path.joinpath(Path(frames_path), f'frame_{frame_idx-1}.png')), frame)

    frames_idx = list(annotations.keys())

    np.random.shuffle(frames_idx)

    train_idx = frames_idx[:int(len(frames_idx)*0.5)]
    test_idx = frames_idx[int(len(frames_idx)*0.5):int(len(frames_idx)*0.75)]
    val_idx = frames_idx[int(len(frames_idx)*0.75):]


    result_path = str(Path.joinpath(Path(__file__).parent, f'./results/week5/task1_1/aic19_{video_seq}_{cam}').absolute())

    train(model_name=model_name,
          results_path=result_path,
          train_idx=train_idx,
          test_idx=test_idx,
          val_idx=val_idx,
          annotations=annotations,
          frames_path=frames_path,
          lr=lr,
          max_it=max_it,
          batch_size=batch_size,
          num_freeze=num_freeze)
"""

if __name__ == "__main__":
    #plot_gt()

    """
    train_single_cam(video_seq='S03',
                cam='c010',
                model_name='COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml',
                lr=0.0001,
                max_it=500,
                batch_size=512,
                num_freeze=1)

    """
    train_seq(train_seqs=['S01', 'S04'],
            test_seqs=['S03'],
            model_name='COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml',
            lr=0.0001,
            max_it=100,
            batch_size=512,
            img_per_batch=16,
            num_freeze=1,
            det='mask_rcnn'
    )
