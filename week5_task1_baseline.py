from pathlib import Path
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.readers.ai_city_reader import resolve_tracks_from_detections, group_by_frame
from src.video import get_frames_from_video, generate_video
from src.metrics.mot_metrics import IDF1Computation
from src.tracking import filter_moving_tracks

import cv2

DATA_DIR = Path('data/AICity_track_data/train')


def task_1_compute_params(distance_thresholds, min_recurrent_tracking, algorithm, detector):
    """ Function to compute the best parametres for the baselines and methodologies we use """
    #  Constants
    video_path = DATA_DIR / 'S03' / 'c010' / 'vdo.avi'
    data_dir = DATA_DIR / 'S03' / 'c010'
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_percentage = 1
    start = 0
    end = int(n_frames * video_percentage)

    # Groundtruth
    gt_reader = AICityChallengeAnnotationReader(str(data_dir / 'gt' / 'gt.txt'))
    gt = gt_reader.get_annotations(classes=['car'])

    # Detections
    det_reader = AICityChallengeAnnotationReader(str(data_dir / 'mtsc' / f'mtsc_{algorithm}_{detector}.txt'))
    dets = det_reader.get_annotations(classes=['car'])
    dets = [dets.get(frame, []) for frame in range(start, end)]
    ordered_dets = []
    for detections in dets:
        ordered_dets.extend(detections)
    all_tracks = list(resolve_tracks_from_detections(ordered_dets).values())

    # Initialize Variables
    y_true = [gt.get(frame, []) for frame in range(start, end)]
    idf1s = []

    for dist_th in distance_thresholds:
        metrics = IDF1Computation()
        y_pred = []

        moving_tracks = filter_moving_tracks(all_tracks, dist_th, min_recurrent_tracking)
        detections = []
        for track in moving_tracks:
            detections.extend(track.tracking)
        detections = group_by_frame(detections)
        
        for frame_idx, current_frame in get_frames_from_video(str(video_path), start_frame=start, end_frame=end):
            frame_detections = []
            for det in detections.get(frame_idx-1, []):
                frame_detections.append(det)
            y_pred.append(frame_detections)
            metrics.add_frame_detections(y_true[frame_idx-1], y_pred[-1])
        summary = metrics.get_computation()
        print(summary)
        idf1s.append(summary['idf1']['metrics']*100)
    print(idf1s)

if __name__ == "__main__":
    distance_thresholds = [500]
    min_recurrent_tracking = 5
    algorithm = 'tc'
    detector = 'mask_rcnn'
    task_1_compute_params(distance_thresholds, min_recurrent_tracking, algorithm, detector)
