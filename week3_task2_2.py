from collections import defaultdict

import numpy as np
import cv2

from src.sort import Sort
from src.metrics.map import mAP
from src.metrics.mot_metrics import IDF1Computation
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.detection import Detection

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class VideoContextManager:

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()


def show_tracks(cap, frame, tracks, colors, detections):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    for d in detections:
        np.random.seed(d.id)
        color = colors.get(d.id, tuple(np.random.randint(0, 256, 3).tolist()))
        colors[d.id] = color
        for bbox in tracks[d.id]:
            tl = (int(bbox[0]), int(bbox[1]))
            br = (int(bbox[2]), int(bbox[3]))
            width = br[0] - tl[0]
            height = br[1] - tl[1]
            center = int(tl[0] + (width / 2)), int(tl[1] + (height / 2))
            cv2.circle(img, center, 5, color, -1)
        cv2.rectangle(img, tl, br, color, thickness=5)

    cv2.imshow('image', cv2.resize(img, (900, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return


def task2_2(det_path, gt_path, video_path, max_age, min_hits,iou_threshold, show):
    with VideoContextManager(video_path=video_path) as cap:
        gt = AICityChallengeAnnotationReader(path=gt_path).get_annotations(classes=['car'])
        dets = AICityChallengeAnnotationReader(path=det_path).get_annotations(classes=['car'])

        tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        tracks = defaultdict(list)
        colors = dict()

        y_true, y_pred = [], []
        idf1_computation = IDF1Computation()
        for frame, detections in dets.items():
            new_detections = tracker.update(np.array([[*detection.bbox, detection.score] for detection in detections]))
            new_detections = [Detection(frame, int(detection[-1]), 'car', *detection[:4]) for detection in new_detections]
            for d in new_detections:
                tracks[d.id].append(d.bbox)
            if show:
                show_tracks(cap, frame, tracks, colors, new_detections)

            gt_frame_detections = gt.get(frame, [])

            idf1_computation.add_frame_detections(gt_frame_detections, new_detections)

            y_true.append(gt_frame_detections)
            y_pred.append(new_detections)

    cv2.destroyAllWindows()

    men_average_precision, prec, rec = mAP(y_true, y_pred, classes=['car'])
    idf1 = idf1_computation.get_computation()
    # print(f'mAP: {men_average_precision:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, IDF1: {idf1:.4f}')
    retstr = f'mAP: {men_average_precision:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, IDF1: {idf1:.4f}'
    return retstr


if __name__ == '__main__':
    paths= ["s03_c010-faster_rcnn.txt",
    "s03_c010-mask_rcnn.txt",
    "s03_c010-retinanet.txt"]
    models = [
        "Faster RCNN",
        "Mask RCNN",
        "RetinaNet"
    ]
    for i, path in enumerate(paths):
        result = task2_2(det_path=path,
                gt_path='data/ai_challenge_s03_c010-full_annotation.xml',
                video_path='data/AICity_data/train/S03/c010/vdo.avi',
                max_age=1,
                min_hits=3,
                iou_threshold = 0.3,
                show=False)
        print(f"{models[i]}: {result}")
        
