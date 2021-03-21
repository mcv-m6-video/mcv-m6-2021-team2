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


def task2_2(det_path='data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt', debug=False):
    """
    Object tracking: tracking with Sort method
    """

    reader = AICityChallengeAnnotationReader(path='data/ai_challenge_s03_c010-full_annotation.xml')
    gt = reader.get_annotations(classes=['car'])
    reader = AICityChallengeAnnotationReader(path=det_path)
    dets = reader.get_annotations(classes=['car'])

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')

    tracker = Sort()
    tracks = defaultdict(list)

    y_true = []
    y_pred = []
    idf1_computation = IDF1Computation()
    for frame in dets.keys():
        detections = dets.get(frame, [])

        new_detections = tracker.update(np.array([[*d.bbox, d.score] for d in detections]))
        new_detections = [Detection(frame, int(d[-1]), 'car', *d[:4]) for d in new_detections]

        y_true.append(gt.get(frame, []))
        y_pred.append(new_detections)

        idf1_computation.add_frame_detections(y_true[-1], y_pred[-1])

        if debug:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            for d in new_detections:
                tracks[d.id].append(d.bbox)
                np.random.seed(d.id)
                color = tuple(np.random.randint(0, 256, 3).tolist())
                for dd in tracks[d.id]:
                    cv2.circle(img, (int((dd[0]+dd[2])/2), int((dd[1]+dd[3])/2)), 5, color, -1)

            cv2.imshow('image', cv2.resize(img, (900, 600)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    ap, prec, rec = mAP(y_true, y_pred, classes=['car'])
    idf1 = idf1_computation.get_computation()
    print(f"AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, IDF1: {idf1:.4f}")


if __name__ == '__main__':
    task2_2(det_path='data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt', debug=True)
