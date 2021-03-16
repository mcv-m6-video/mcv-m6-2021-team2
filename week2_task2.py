from pathlib import Path
import cv2
import numpy as np
from random import randint

from src.metrics.map import mAP
from src.readers.ai_city_reader import read_xml, read_txt
from src.annotation import Annotation
from src.utils.generate_video import generate_video
from src.utils.plot import plot_img
from src.video import generate_model, get_frames_from_video, generate_video_from_frames
from src.segmentation import (
    apply_roi_on_segmented_frames,
    remove_noise_on_segmented_frames,
    find_objects_in_segmented_frames
)

# -- DIRECTORIES -- #
RESULTS = Path('results/week2')
VIDEO_PATH = Path('data/AICity_data/train/S03/c010/vdo.avi')

def task_2():
	background, variance = generate_model(str(VIDEO_PATH), 0.25)
	plot_img(background, cmap='gray', title='model_mean_background', save_root=RESULTS)
	plot_img(variance, cmap='afmhot', title='var_model_background', save_root=RESULTS)


if __name__ == "__main__":
    task_2()