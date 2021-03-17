from pathlib import Path
import cv2
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from glob import glob

from src.metrics.map import mAP
from src.readers.ai_city_reader import read_xml, read_txt
from src.annotation import Annotation
from src.utils.generate_video import generate_video
from src.utils.plot import plot_img
from src.video import generate_model, get_frames_from_video, generate_video_from_frames
from src.video import get_multi_model
from src.segmentation import (
    single_adapted_gaussian_segmentation,
    apply_roi_on_segmented_frames,
    remove_noise_on_segmented_frames,
    find_objects_in_segmented_frames
)

# -- DIRECTORIES -- #
RESULTS = Path('results/week2')
GT = Path('data/ai_challenge_s03_c010-full_annotation.xml')
VIDEO_PATH = Path('data/AICity_data/train/S03/c010/vdo.avi')
ROI = Path('data/AICity_data/train/S03/c010/roi.jpg')
FRAMES_LOCATION = Path('data/frames')


def task_2_1():
    split_percentage = 0.25
    background, variance = generate_model(str(VIDEO_PATH), split_percentage)
    frames, _, _ = get_frames_from_video(VIDEO_PATH)
    frames = [(int(len(frames)*split_percentage) + frame_idx, frame ) for frame_idx, frame in enumerate(frames[int(len(frames)*split_percentage):])]

    #plot_img(background, cmap='gray', title='model_mean_background', save_root=RESULTS)
    #plot_img(variance, cmap='afmhot', title='var_model_background', save_root=RESULTS)

    gt_annotations = read_xml(GT, include_parked=False)
    rhos = [0.01, 0.02, 0.25, 0.03, 0.035, 0.04]
    alpha = 6.0
    kernel_dim = (5,5)
    region_to_detect = (120, 120)

    for k,rho in enumerate(rhos):
        print(f'Run {k}/{len(rhos)}: alpha: {alpha} - rho: {rho} - kernel: {kernel_dim} - region to detect: {region_to_detect} ')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dim)

        segmented_frames = single_adapted_gaussian_segmentation(background, variance, frames, alpha, rho)
        segmented_frames = apply_roi_on_segmented_frames(segmented_frames, str(ROI))
        segmented_frames = remove_noise_on_segmented_frames(segmented_frames, kernel)

        annotations = find_objects_in_segmented_frames(segmented_frames, region_to_detect)

        mapp, _, _, mious_per_class = mAP(annotations, gt_annotations, ['car'], False)

        print('mAP: ', mapp)
        print('-'*10)

def task_2_2():
    split_percentage = 0.25
    background, variance = generate_model(str(VIDEO_PATH), split_percentage)
    frames, _, _ = get_frames_from_video(VIDEO_PATH)
    frames = [(int(len(frames)*split_percentage) + frame_idx, frame ) for frame_idx, frame in enumerate(frames[int(len(frames)*split_percentage):])]

    #plot_img(background, cmap='gray', title='model_mean_background', save_root=RESULTS)
    #plot_img(variance, cmap='afmhot', title='var_model_background', save_root=RESULTS)

    gt_annotations = read_xml(GT, include_parked=False)

    alphas = [4.0, 5.0, 6.0]
    rhos = [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
    kernels = [(3,3), (5,5)]
    regions_to_detect = [(120, 120), (200, 150), (150, 200)]

    for n in range(0, 10):
        alpha = alphas[randint(0, len(alphas)-1)]
        rho = rhos[randint(0, len(rhos)-1)]
        kernel_dim = kernels[randint(0, len(kernels)-1)]
        region_to_detect = regions_to_detect[randint(0, len(regions_to_detect)-1)]

        print(f'Run {n}/10: alpha: {alpha} - rho: {rho} - kernel: {kernel_dim} - region to detect: {region_to_detect} ')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_dim)

        segmented_frames = single_adapted_gaussian_segmentation(background, variance, frames, alpha, rho)
        segmented_frames = apply_roi_on_segmented_frames(segmented_frames, str(ROI))
        segmented_frames = remove_noise_on_segmented_frames(segmented_frames, kernel)

        annotations = find_objects_in_segmented_frames(segmented_frames, region_to_detect)

        mapp, _, _, mious_per_class = mAP(annotations, gt_annotations, ['car'], False)

        print('mAP: ', mapp)
        print('-'*10)

def task_2_1_video():
    split_percentage = 0.25
    background, variance = generate_model(str(VIDEO_PATH), split_percentage)
    #paths = sorted(glob('data/frames/*jpg'))
    #frames = [cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) for img_path in paths]
    frames, _, _ = get_frames_from_video(VIDEO_PATH)
    frames = [(int(len(frames)*split_percentage) + frame_idx, frame ) for frame_idx, frame in enumerate(frames[int(len(frames)*split_percentage):])]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    gt_annotations = read_xml(GT, include_parked=False)

    segmented_frames = single_adapted_gaussian_segmentation(background, variance, frames, 9, 0.04)
    segmented_frames = apply_roi_on_segmented_frames(segmented_frames, str(ROI))
    segmented_frames = remove_noise_on_segmented_frames(segmented_frames, kernel)

    annotations = find_objects_in_segmented_frames(segmented_frames, (5,5))

    generate_video_from_frames(str(RESULTS/'video_alpha_9_rho0.04.gif'), 0.6, segmented_frames[:200])

    mapp, _, _, mious_per_class = mAP(annotations, gt_annotations, ['car'], False)
    print('mAP: ', mapp)

def task_4():
    split_percentage = 0.25
    background, variance = get_multi_model(str(VIDEO_PATH), split_percentage, colorspace='rgb')
    frames, _, _ = get_frames_from_video(VIDEO_PATH, False, colorspace='rgb')
    frames = [(int(len(frames)*split_percentage) + frame_idx, frame ) for frame_idx, frame in enumerate(frames[int(len(frames)*split_percentage):])]
    print(background.shape, variance.shape)

    plot_img(background, title='model_mean_background_3', save_root=RESULTS)
    plot_img(variance, title='var_model_background_3', save_root=RESULTS)
    """
    frames, _, _ = get_frames_from_video(VIDEO_PATH)
    frames = [(int(len(frames)*split_percentage) + frame_idx, frame ) for frame_idx, frame in enumerate(frames[int(len(frames)*split_percentage):])]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    gt_annotations = read_xml(GT, include_parked=False)

    segmented_frames = single_adapted_gaussian_segmentation(background, variance, frames, 6, 0.02)
    segmented_frames = apply_roi_on_segmented_frames(segmented_frames, str(ROI))
    segmented_frames = remove_noise_on_segmented_frames(segmented_frames, kernel)

    annotations = find_objects_in_segmented_frames(segmented_frames, (5,5))

    generate_video_from_frames(str(RESULTS/'video_alpha_9_rho0.04.gif'), 0.6, segmented_frames[:200])

    mapp, _, _, mious_per_class = mAP(annotations, gt_annotations, ['car'], False)
    print('mAP: ', mapp)
    """

if __name__ == "__main__":
    task_4()