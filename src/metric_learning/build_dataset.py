from typing import List, NoReturn
import os
from pathlib import Path
import cv2
import numpy as np
import shutil

from src.video import get_frames_from_video
from src.detection import Detection
from src.readers.ai_city_reader import AICityChallengeAnnotationReader


""" TO REVIEW
def analyze_data(main_path, debug=False):
    areas = []
    heights = []
    widths = []
    for gt_file in glob.glob(os.path.join(main_path, "*", "*", "gt", "gt.txt")):
        video_path = gt_file.replace("gt\\gt.txt", "vdo.avi")
        cap = cv2.VideoCapture(video_path)
        gt = parse_annotations_from_txt(gt_file)
        for frame, detections in gt.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            for det in detections:
                areas.append(det.area)
                heights.append(det.height)
                widths.append(det.width)
                if debug:
                    cv2.rectangle(img, (int(det.xtl), int(det.ytl)), (int(det.xbr), int(det.ybr)), (0, 255, 0), 2)
            if debug:
                cv2.imshow('result', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    plt.hist(areas, 100, facecolor='blue')
    plt.show()
    print("Min area", min(areas), " Min width", min(widths), " Min height", min(heights))


def downsample(img, max_height, max_width):
    height, width = img.shape[:2]
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img
"""

def crop_car_from_frame(detection: Detection,
                            frame: np.array,
                            img_size: int) -> np.array:
    car = frame[int(detection.ytl):int(detection.ybr),int(detection.xtl):int(detection.xbr)]
    return cv2.resize(car, (img_size, img_size))


def generate_sequence(seq: str,
                      dataset_path: str,
                      output_path: str,
                      img_size: int) -> NoReturn:
    for cam in Path(Path.joinpath(Path(dataset_path), seq)).iterdir():
        if cam.is_dir():
            output_dataset_path = str(Path.joinpath(Path(__file__).parent, f'../../data/ml_database/{seq}/{cam.name}'))

            video_path = str(Path.joinpath(cam, f'./vdo.avi'))

            gt_path = str(Path.joinpath(cam, f'./gt/gt.txt'))
            gt_annotations = AICityChallengeAnnotationReader(gt_path).get_annotations(classes=['car'])

            print(f'Seq {seq} - Cam {cam.name}')
            for frame_idx, frame in get_frames_from_video(video_path):
                if (frame_idx - 1) not in gt_annotations:
                    continue

                for det in gt_annotations[frame_idx - 1]:
                    if det.width >= img_size and det.height >= img_size:
                        crop = crop_car_from_frame(det, frame, img_size)

                        output_crop_path = str(Path.joinpath(Path(output_path), f'{det.id}/{cam.name}_{frame_idx - 1}.png'))
                        os.makedirs(str(Path.joinpath(Path(output_path), str(det.id))), exist_ok=True)

                        cv2.imwrite(output_crop_path, crop)

def generate_metric_learning_database(dataset_path: str,
                                      train_seq: List[str],
                                      test_seq: List[str],
                                      img_size: int) -> NoReturn:

    if Path.joinpath(Path(__file__).parent, f'../../data/ml_database').exists():
        shutil.rmtree(str(Path.joinpath(Path(__file__).parent, f'../../data/ml_database')))

    for seq in train_seq:
        output_path = str(Path.joinpath(Path(__file__).parent, '../../data/ml_database/train'))
        generate_sequence(seq=seq,
                          dataset_path=dataset_path,
                          output_path=output_path,
                          img_size=img_size)
    for seq in test_seq:
        output_path = str(Path.joinpath(Path(__file__).parent, '../../data/ml_database/test'))
        generate_sequence(seq=seq,
                          dataset_path=dataset_path,
                          output_path=output_path,
                          img_size=img_size)


