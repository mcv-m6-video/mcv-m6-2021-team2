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
    background, variance = generate_model(video_path, percentatge_frame_to_use)

    for alpha in [2.0, 3.0, 4.0, 5.0]:
        segmented_frames = single_gaussian_segmentation(background, variance, frames[:100], alpha)

        #cv2.imwrite('./results/week2/example_before_roi.png', segmented_frames[-1][1])
        segmented_frames = apply_roi_on_segmented_frames(segmented_frames)
        #cv2.imwrite('./results/week2/example_after_roi.png', segmented_frames[-1][1])

        #cv2.imwrite('./results/week2/example_before_opening.png', segmented_frames[-1][1])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        segmented_frames = remove_noise_on_segmented_frames(segmented_frames, kernel)
        #cv2.imwrite('./results/week2/example_after_opening.png', segmented_frames[-1][1])

        annotations = find_objects_in_segmented_frames(segmented_frames, (20, 20))

        frames_dict = {}
        for frame_idx, frame in segmented_frames:
            frames_dict[frame_idx] = frame

        frame_idx = annotations[-1].frame
        frame = frames_dict[frame_idx]

        """
        cv2.imwrite('./results/week2/example_before_find_bbox.png', frames_dict[frame_idx])
        for annon in annotations:
            if annon.frame == frame_idx:
                frame = cv2.rectangle(frame, (int(annon.left), int(annon.top)), (int(annon.width), int(annon.height)), (255, 0, 0), 2)
        cv2.imwrite('./results/week2/example_after_find_bbox.png', frame)
        """

        gt_path = str(Path.joinpath(Path(__file__).parent, './data/s03_c010-annotation.xml'))
        gt_annotations = read_xml(gt_path, include_parked=True)

        """
        example_path = str(Path.joinpath(Path(__file__).parent, './data/s03_c010-ssd512.txt'))
        example_annotations = read_txt(example_path)

        print('SSD 512: ',mAP(example_annotations, gt_annotations, ['car'], True)[0])
        """

        mapp, _, _, mious_per_class = mAP(annotations, gt_annotations, ['car'], False)
        print('Segmentation: ', mapp)

        frames_idx = set()
        for annon in annotations:
            frames_idx.add(annon.frame)

        #generate_video(video_path, list(frames_idx), gt_annotations, annotations, 'test', 'example.gif')

        mious_per_frame = []
        for frame_idx in list(frames_idx):
            miou = mious_per_class['car'][frame_idx]
            mious_per_frame.append(miou)


        generate_video(video_path, list(frames_idx), gt_annotations, annotations, 'test', 'example2.gif', mious_per_frame)


task1_2()

"""
cv2.imwrite('before_roi_result.png', result[-1][1])
rois = apply_roi_on_segmented_frames(result)
cv2.imwrite('after_roi_result.png', rois[-1][1])
opens = remove_noise_on_segmented_frames(rois, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
cv2.imwrite('open_result.png', opens[-1][1])
#generate_video_from_frames('example.gif', result[:100])
"""