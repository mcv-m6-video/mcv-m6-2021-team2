from typing import List, Tuple, NoReturn, OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import cv2
import os

def get_frames_from_video(video_path: str,
                          start_frame: int = 0,
                          end_frame: int = np.inf) -> Tuple[int, np.array]:

    if not Path(video_path).exists:
        raise FileNotFoundError(f'Video path not found: {video_path}.')

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame < 0:
        raise ValueError(f"Start frame ({start_frame}) should be greater than 0.")
    if end_frame == np.inf:
        end_frame = frame_count
    elif end_frame > frame_count:
        raise ValueError(f"End frame ({end_frame}) is greater than {frame_count} which is the number of video frames.")

    for frame_idx in tqdm(range(start_frame, end_frame, 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        has_frames, frame = cap.read()

        if has_frames:
            yield (frame_idx + 1, frame)

def get_video_length(video_path: str) -> int:
    if not Path(video_path).exists:
        raise FileNotFoundError(f'Video path not found: {video_path}.')

    cap = cv2.VideoCapture(video_path)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return num_frames

def generate_video(video_path: str,
                   output_path: str,
                   predictions: OrderedDict,
                   gt: OrderedDict,
                   title: str,
                   start_frame: int = 0,
                   end_frame: int = np.inf) -> NoReturn:

    if not Path(video_path).exists:
        raise FileNotFoundError(f'Video path not found: {video_path}.')

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame < 0:
        raise ValueError(f"Start frame ({start_frame}) should be greater than 0.")
    if end_frame == np.inf:
        end_frame = frame_count
    elif end_frame > frame_count:
        raise ValueError(f"End frame ({end_frame}) is greater than {frame_count} which is the number of video frames.")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    image = ax.imshow(np.zeros((height, width)))
    artists = [image]

    frames_index = list(range(start_frame, end_frame, 1))

    def update(i):
        print(f'Processing frame {frames_index[i] + 1}/{end_frame} ...')

        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_index[i])

        ret, img = cap.read()

        detections_on_frame = predictions.get(frames_index[i], [])
        for det in detections_on_frame:
            cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 0, 255), 2)

        gt_on_frame = gt.get(frames_index[i], [])
        for det in gt_on_frame:
            cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 2)

        artists[0].set_data(img[:, :, ::-1])

        return artists

    ani = FuncAnimation(fig, update, len(frames_index), interval=2, blit=True)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=[patches.Patch(color="green", label="GT"), patches.Patch(color="red", label="Pred")])

    fig.suptitle(title)
    ani.save(output_path, writer='pillow')

    cap.release()
