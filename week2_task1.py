from pathlib import Path
import cv2
import numpy as np
from random import randint

from src.metrics.map import mAP
from src.readers.ai_city_reader import read_xml, read_txt
from src.annotation import Annotation
from src.utils.generate_video import generate_video

from src.video import generate_model, get_frames_from_video, generate_video_from_frames
from src.segmentation import (
    single_gaussian_segmentation,
    apply_roi_on_segmented_frames,
    remove_noise_on_segmented_frames,
    find_objects_in_segmented_frames
)

percentatge_frame_to_use = 0.25
video_path = str(Path.joinpath(Path(__file__).parent, './data/vdo.avi'))

frames, _, _ = get_frames_from_video(video_path)
frames = [(int(len(frames)*percentatge_frame_to_use) + frame_idx, frame ) for frame_idx, frame in enumerate(frames[int(len(frames)*percentatge_frame_to_use):])]


def task1_1():
    background, variance = generate_model(video_path, percentatge_frame_to_use)
    cv2.imwrite("background.png", background)
    cv2.imwrite("variance.png", variance)

    for alpha in [1.0, 2.0, 4.0, 5.0]:
        segmented_frames = single_gaussian_segmentation(background, variance, frames[:100], alpha)
        generate_video_from_frames(f'./results/week2/video_alpha_{alpha}.gif', 0.6, segmented_frames)

def task1_2():
    gt_path = str(Path.joinpath(Path(__file__).parent, './data/s03_c010-annotation.xml'))
    gt_annotations = read_xml(gt_path, include_parked=True)

    # Compute the mean of car size

    background, variance = generate_model(video_path, percentatge_frame_to_use)
    alphas = [2.0, 2.5, 3.0, 3,5, 4.0, 4.5, 5.0, 5.0, 5.5, 6.0]
    kernels = [(5,5), (6,6), (2,2), (4,4), (7,7), (1,1), (3,3)]
    regions_to_detect = [(100,100), (50, 50), (100, 75), (150, 200), (150, 150), (125, 100), (75, 100)]
    include_parked_cars = [True, False]

    for n in range(0, 30):
        alpha = alphas[randint(0, len(alphas)-1)]
        kernel_dim = kernels[randint(0, len(kernels)-1)]
        region_to_detect = regions_to_detect[randint(0, len(regions_to_detect)-1)]
        include_parked = include_parked_cars[randint(0, 1)]

        alpha=6.0
        kernel_dim=(4,4)
        region_to_detect=(100,100)
        include_parked=False

        print(f'Run {n}/10: alpha {alpha} - kernel {kernel_dim} - region to detect {region_to_detect} - include parked {include_parked}')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dim)

        segmented_frames = single_gaussian_segmentation(background, variance, frames, alpha)
        segmented_frames = apply_roi_on_segmented_frames(segmented_frames)
        segmented_frames = remove_noise_on_segmented_frames(segmented_frames, kernel)

        annotations = find_objects_in_segmented_frames(segmented_frames, region_to_detect)

        gt_annotations = read_xml(gt_path, include_parked=include_parked)
        mapp, _, _, mious_per_class = mAP(annotations, gt_annotations, ['car'], False)

        print('mAP: ', mapp)

        frames_idx = set()
        for frame_idx, _ in frames:
            frames_idx.add(frame_idx)
        frames_idx = list(frames_idx)[:200]
        #generate_video(video_path, list(frames_idx), gt_annotations, annotations, 'test', 'example.gif')

        mious_per_frame = []
        for frame_idx in frames_idx:
            if frame_idx in mious_per_class['car']:
                miou = mious_per_class['car'][frame_idx]
            else:
                miou = 0.0
            mious_per_frame.append(miou)


        generate_video(video_path, frames_idx, gt_annotations, annotations, 'test', 'example2.gif', mious_per_frame)
        return

task1_2()

"""
cv2.imwrite('before_roi_result.png', result[-1][1])
rois = apply_roi_on_segmented_frames(result)
cv2.imwrite('after_roi_result.png', rois[-1][1])
opens = remove_noise_on_segmented_frames(rois, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
cv2.imwrite('open_result.png', opens[-1][1])
#generate_video_from_frames('example.gif', result[:100])
"""