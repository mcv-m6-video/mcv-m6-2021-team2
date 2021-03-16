from typing import List, Tuple, NoReturn
from pathlib import Path
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pygifsicle import optimize
import logging
import numpy as np
import imageio
import cv2
import shutil
import pickle
import os

def get_frames_from_video(path: str,
                          grayscale: bool = True) -> Tuple[List[np.array], int, int]:

    if not Path(path).exists:
        raise FileNotFoundError(f'Video path not found: {path}.')

    logging.info(f"Processing video from: {path} ...")

    if Path('video_frames.pickle').exists():
        with open('video_frames.pickle', 'rb') as f:
            return pickle.load(f)

    cap = cv2.VideoCapture(path)

    video_frames = []
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    has_frames = True

    while has_frames:
        has_frames, frame = cap.read()

        if has_frames:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            video_frames.append(frame)

    cap.release()

    logging.info(f"Total frames processed: {len(video_frames)} ...")

    with open('video_frames.pickle', 'wb') as f:
        pickle.dump((video_frames, frame_width, frame_height), f)

    return video_frames, frame_width, frame_height

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