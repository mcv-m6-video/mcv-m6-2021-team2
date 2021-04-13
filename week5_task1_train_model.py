from pathlib import Path
from src.readers.ai_city_reader import parse_annotations_from_txt
from src.metrics.map import mAP

import cv2
import os
import numpy as np

from src.detectron.train import train
from src.detectron.inference import inference_from_trained_model
from src.video import get_frames_from_video, generate_video, get_video_lenght


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
    video_seq_path = str(Path.joinpath(Path(__file__).parent, f'./aic19-track1-mtmc-train/train/S03/c010/gt/gt.txt'))
    annotations = parse_annotations_from_txt(video_seq_path)

    i = 0
    for frame_idx, dets in annotations.items():
        frame_path = str(Path.joinpath(Path(__file__).parent, f'./aic19_frames/S03_all_cams/frame_{frame_idx}.png'))
        img = cv2.imread(frame_path)

        for det in dets:
            cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 2)
        cv2.imwrite(f'./tmp/frame_{frame_idx}.png', img)
        i = i + 1
        if i == 5:
            return


def train_seq(video_seq: str,
              model_name: str,
              lr: float,
              max_it: int,
              batch_size: int,
              num_freeze: int):

    video_seq_cam = {
        'S03' : ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    }

    video_seq_path = str(Path.joinpath(Path(__file__).parent,
                   f'./aic19-track1-mtmc-train/train/{video_seq}'))

    if not Path(video_seq_path).exists():
        raise ValueError(f'The cam path: {video_seq_path} does not exist.')

    video_seq_annotations = {}

    offset = 0
    for cam in video_seq_cam[video_seq]:
        cam_gt_path = str(Path.joinpath(Path(video_seq_path), f'./{cam}/gt/gt.txt'))
        cam_annotations = parse_annotations_from_txt(cam_gt_path)

        for frame_idx, annotations in cam_annotations.items():
            for annotation in annotations:
                annotation.cam = cam
                annotation.seq = video_seq
                annotation.id = offset + frame_idx
            video_seq_annotations[offset + frame_idx] = annotations
        offset = offset + get_video_lenght(str(Path.joinpath(Path(video_seq_path), f'./{cam}/vdo.avi')))

    video_seq_frames_path = str(Path.joinpath(Path(__file__).parent, f'./aic19_frames/{video_seq}_all_cams').absolute())

    if not Path(video_seq_frames_path).exists():
        os.makedirs(video_seq_frames_path, exist_ok=True)

        offset = 0
        for cam in video_seq_cam[video_seq]:
            video_path = str(Path.joinpath(Path(video_seq_path), f'./{cam}/vdo.avi'))

            for frame_idx, frame in get_frames_from_video(video_path):
                if (offset + frame_idx - 1) in video_seq_annotations:
                    cv2.imwrite(str(Path.joinpath(Path(video_seq_frames_path), f'frame_{offset+frame_idx-1}.png')), frame)

            offset = offset + get_video_lenght(video_path)

    result_path = str(Path.joinpath(Path(__file__).parent, f'./results/week5/task1_1/aic19_{video_seq}').absolute())

    frames_idx = list(video_seq_annotations.keys())
    #np.random.shuffle(frames_idx)

    train_idx = frames_idx[:int(len(frames_idx)*0.25)]
    test_idx = frames_idx[int(len(frames_idx)*0.25):int(len(frames_idx)*0.5)]
    val_idx = frames_idx[int(len(frames_idx)*0.5):]

    train(model_name=model_name,
          results_path=result_path,
          train_idx=train_idx,
          test_idx=test_idx,
          val_idx=val_idx,
          annotations=video_seq_annotations,
          frames_path=video_seq_frames_path,
          lr=lr,
          max_it=max_it,
          batch_size=batch_size,
          num_freeze=num_freeze)

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

    frames_path = str(Path.joinpath(Path(__file__).parent, f'./aic19_frames/{video_seq}/{cam}'))
    if not Path(frames_path).exists():
        os.makedirs(frames_path)

        for frame_idx, frame in get_frames_from_video(video_path):
            cv2.imwrite(str(Path.joinpath(Path(frames_path), f'frame{frame_idx-1}.png')), frame)

    video_lenght = get_video_lenght(video_path)
    frames_idx = list(range(0, video_lenght))
    np.random.shuffle(frames_idx)

    train_idx = frames_idx[:int(video_lenght*0.25)]
    test_idx = frames_idx[int(video_lenght*0.25):]

    cam_gt_path = str(Path.joinpath(Path(cam_path), './gt/gt.txt'))

    gt_reader = AICityChallengeAnnotationReader(cam_gt_path)
    annotations = gt_reader.get_annotations(classes=['car'])

    result_path = str(Path.joinpath(Path(__file__).parent, f'./results/week5/task1_1/aic19_{video_seq}_{cam}').absolute())

    train(video_seq=video_seq,
          cam=cam,
          model_name=model_name,
          results_path=result_path,
          train_idx=train_idx,
          test_idx=test_idx,
          annotations=annotations,
          frames_path=frames_path,
          lr=lr,
          max_it=max_it,
          batch_size=batch_size,
          num_freeze=num_freeze)


if __name__ == "__main__":
    train_seq(video_seq='S03',
                model_name='COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml',
                lr=0.0001,
                max_it=100,
                batch_size=512,
                num_freeze=1)