from typing import List, Tuple, NoReturn, OrderedDict

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from pygifsicle import optimize
import logging
import numpy as np
import imageio
import cv2

def get_frames_from_video(video_path: str,
                          colorspace: str = 'rgb',
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

    logging.debug(f'Processing video from frames: {start_frame} to {end_frame} ...')
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        has_frames, frame = cap.read()

        logging.debug(f'Frame: {frame_idx+1}/{end_frame}.')

        if has_frames:
            if colorspace == 'gray':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif colorspace == 'rgb':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif colorspace == 'hsv':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            elif colorspace == 'lab':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            else:
                raise NotImplementedError(f'The colorspace {colorspace} is not supported.')

            yield (frame_idx+1, frame)
            frame_idx += 1
        else:
            logging.error(f'The video doesn\'t have the frame {frame_idx+1}.')

    cap.release()
    yield (0, None)

def get_video_lengh(video_path: str) -> int:
    if not Path(video_path).exists:
        raise FileNotFoundError(f'Video path not found: {video_path}.')

    cap = cv2.VideoCapture(video_path)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return num_frames


def get_frame_from_video(video_path: str,
                         frame: int,
                         colorspace: str = 'rgb') -> np.array:

    if not Path(video_path).exists:
        raise FileNotFoundError(f'Video path not found: {video_path}.')

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame < 0:
        raise ValueError(f"Frame ({frame}) should be greater than 0.")
    elif frame > frame_count:
        raise ValueError(f"Frame ({frame}) is greater than {frame_count} which is the number of video frames.")

    logging.debug(f'Processing video frame: {frame} ...')

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    has_frames, frame = cap.read()

    if has_frames:
        if colorspace == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif colorspace == 'rgb':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif colorspace == 'hsv':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif colorspace == 'lab':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        else:
            raise NotImplementedError(f'The colorspace {colorspace} is not supported.')

    cap.release()

    return frame


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
    logging.getLogger('matplotlib.font_manager').disabled = True

    if start_frame < 0:
        raise ValueError(f"Start frame ({start_frame}) should be greater than 0.")
    if end_frame == np.inf:
        end_frame = frame_count
    elif end_frame > frame_count:
        raise ValueError(f"End frame ({end_frame}) is greater than {frame_count} which is the number of video frames.")

    logging.debug(f'Processing video from frames: {start_frame} to {end_frame} ...')

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    image = ax.imshow(np.zeros((height, width)))
    artists = [image]

    frames_index = list(range(start_frame, end_frame, 1))

    def update(i):
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
    ani.save(output_path, writer='imagemagick')

    cap.release()


def generate_model(video_path: str,
                   perctatge_use_frames: float,
                   grayscale: bool = True) -> Tuple[np.array, np.array]:

    video_frames, frame_width, frame_height = get_frames_from_video(video_path)

    if perctatge_use_frames < 0.0 or perctatge_use_frames > 1.0:
        raise ValueError("The percentatge use of frames should be [0,1].")

    num_frames_to_use = int(len(video_frames)*perctatge_use_frames)

    mean_model_background = np.zeros((frame_height, frame_width))
    variance_model_background = np.zeros((frame_height, frame_width))

    if num_frames_to_use > 0:
        mean_model_background = np.mean(video_frames[:num_frames_to_use], axis=0)
        variance_model_background = np.std(video_frames[:num_frames_to_use], axis=0)

    return mean_model_background, variance_model_background

def get_multi_model(video_path: str,
                   perctatge_use_frames: float,
                   colorspace: str = 'rgb') -> Tuple[np.array, np.array]:
    video_frames, frame_width, frame_height = get_frames_from_video(video_path, False, colorspace=colorspace)

    if perctatge_use_frames < 0.0 or perctatge_use_frames > 1.0:
        raise ValueError("The percentatge use of frames should be [0,1].")

    num_frames_to_use = int(len(video_frames)*perctatge_use_frames)

    mean_model_background = np.zeros((frame_height, frame_width, 3))
    variance_model_background = np.zeros((frame_height, frame_width, 3))

    if num_frames_to_use > 0:
        mean_model_background = np.mean(video_frames[:num_frames_to_use], axis=0)
        variance_model_background = np.std(video_frames[:num_frames_to_use], axis=0)

    return mean_model_background, variance_model_background


def generate_video_from_frames(output_path: str,
                               scale: float,
                               frames: List[Tuple[int, np.array]]) -> NoReturn:

    dim = (int(frames[-1][1].shape[1] * scale), int(frames[-1][1].shape[0] * scale))

    resized_frames = []
    for _, frame in frames:
        frame = frame.astype('uint8')
        resized_frames.append(
            cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        )

    imageio.mimsave(output_path, resized_frames)
    optimize(output_path)