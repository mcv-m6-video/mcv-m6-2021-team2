from typing import Tuple, List, NoReturn, Dict
from pathlib import Path
from skimage.morphology import opening, square
import numpy as np
import cv2

from src.annotation import Annotation

def single_gaussian_segmentation(mean_model_background: np.array,
                                 variance_model_background: np.array,
                                 frames: List[Tuple[int, np.array]],
                                 alpha: float) -> List[Tuple[int, np.array]]:

    frame_shape = frames[-1][1].shape
    segmented_frames = []

    for i, (frame_idx, frame) in enumerate(frames):
        segmented_frame = np.zeros((frame.shape))
        segmented_frame[np.abs(frame - mean_model_background) >= alpha * (variance_model_background + 2)] = 255
        segmented_frames.append((frame_idx, np.ascontiguousarray(segmented_frame).astype("uint8")))

    return segmented_frames

def single_adapted_gaussian_segmentation(mean_model_background: np.array,
                                 variance_model_background: np.array,
                                 frames: List[Tuple[int, np.array]],
                                 alpha: float,
                                 rho: float) -> List[Tuple[int, np.array]]:

    frames_shape = frames[-1][1].shape
    foreground_frames = []

    for i, (frame_idx, frame) in enumerate(frames):
        fg = np.abs(frame-mean_model_background) >= (alpha * (variance_model_background + 2))
        bg = ~fg
        mean_model_background[bg] = rho * frame[bg] + (1 - rho) * mean_model_background[bg]
        variance_model_background[bg] = np.sqrt(
            rho * np.power(frame[bg] - mean_model_background[bg], 2) + (1-rho) * np.power(variance_model_background[bg], 2)
        )
        foreground_frames.append((frame_idx, (fg*255).astype(np.uint8)))

    return foreground_frames

def apply_roi_on_segmented_frames(frames: List[Tuple[int, np.array]], roi_path: str) -> List[Tuple[int, np.array]]:
    roi = cv2.imread(roi_path, cv2.COLOR_BGR2GRAY)

    for i, (frame_idx, frame) in enumerate(frames):
        frames[i] = (frame_idx, cv2.bitwise_and(frame, frame, mask=roi))

    return frames

def remove_noise_on_segmented_frames(frames: List[Tuple[int, np.array]],
                                     kernel: np.array) -> List[Tuple[int, np.array]]:

    for i, (frame_idx, frame) in enumerate(frames):
        # To close little holes in the masks
        mask = cv2.morphologyEx(frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # To remove noise around the masks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        frames[i] = (frame_idx, mask)

    return frames

def find_objects_in_segmented_frames(frames: List[Tuple[int, np.array]],
                                     box_min_size: Tuple) -> List[Annotation]:

    annnotations = []

    for i, (frame_idx, frame) in enumerate(frames):
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)

            if width > box_min_size[0] and height > box_min_size[1]:
                annnotations.append(Annotation(
                    frame=frame_idx,
                    left=left,
                    top=top,
                    width=left + width,
                    height=top + height,
                    label='car',
                ))

    return annnotations