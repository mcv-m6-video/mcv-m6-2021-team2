import numpy as np
import cv2
import imageio

from pathlib import Path
from src.metrics.mot_metrics import IDF1Computation
from src.metrics.map import mAP
from src.video import get_frames_from_video
from src.tracking import MaxOverlapTracker, MaxOverlapTrackerEfficient
from src.readers.ai_city_reader import AICityChallengeAnnotationReader


RESULTS_DIR = Path('Results/week3')
VIDEO_PATH = Path('data/AICity_data/train/S03/c010/vdo.avi')

def task_2_1(prediction_path, efficient=True):
    # Read groundtruth xml detections
    gt_reader = AICityChallengeAnnotationReader(Path('data/ai_challenge_s03_c010-full_annotation.xml'))
    detection_reader = AICityChallengeAnnotationReader(prediction_path)

    gt_detections = gt_reader.get_annotations(classes=['car'])
    pred_detections = detection_reader.get_annotations(classes=['car'])

    # Initialize Variables
    tracker = MaxOverlapTrackerEfficient() if efficient else MaxOverlapTracker()
    tracks = []
    y_gt = []
    y_pred = []
    metrics = IDF1Computation()
    eff = '_eff' if efficient else ''

    # Start Video
    writer = imageio.get_writer(str(RESULTS_DIR / f'task_2_1_{prediction_path.stem}{eff}.gif'), fps=10)

    # For each frame, compute the detected tracks by maximum overlaping
    for frame_idx, frame in get_frames_from_video(str(VIDEO_PATH), colorspace='rgb', start_frame=0):
        # Get detections from current frame
        if frame is not None:
            current_gt_det = gt_detections[frame_idx-1]
            current_pred_dt = pred_detections[frame_idx-1]

            # Compute tracks from the current frame
            tracks, tracks_on_frame = tracker.track_by_max_overlap(tracks, current_pred_dt)

            # Write detected information onto the video
            for track in tracks_on_frame:
                detection = track.previous_detection
                cv2.rectangle(frame, (int(detection.xtl), int(detection.ytl)), (int(detection.xbr), int(detection.ybr)), track.color, 2)
                cv2.rectangle(frame, (int(detection.xtl), int(detection.ytl)), (int(detection.xbr), int(detection.ytl) - 15), track.color, -2)
                cv2.putText(frame, str(detection.id), (int(detection.xtl), int(detection.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                for tt in track.tracking:
                    cv2.circle(frame, tt.center, 5, track.color, -1)

            # Update Variables and Metrics
            y_gt.append(current_gt_det)
            y_pred.append(current_pred_dt)
            metrics.add_frame_detections(y_gt[-1], y_pred[-1])

            # Resize video to occupy less space
            writer.append_data(cv2.resize(frame, (720, 470)))
            print(f' ---- Frame: {frame_idx} Processed. --- ')

    # Tracking done, compute now metrics and print results
    writer.close()
    ap, prec, rec = mAP(y_gt, y_pred, classes=['car'], sort_method='score')
    print(f'AP: {ap:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, IDF1: {metrics.get_computation()}')

if __name__ == '__main__':
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    task_2_1(Path('data/AICity_data/train/S03/c010/det/s03_c010-retinanet.txt'), efficient=True)
