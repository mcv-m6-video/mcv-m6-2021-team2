from pathlib import Path
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.readers.ai_city_reader import resolve_tracks_from_detections, group_by_frame
from src.video import get_frames_from_video, generate_video, get_video_length
from src.metrics.mot_metrics import IDF1Computation
from src.tracking import filter_moving_tracks

import numpy as np
import cv2
import pickle

DATA_DIR = Path('data/AICity_track_data/train')
RESULTS_DIR = Path('results/week5/')

def task_1_find_best_baseline(distance_thresholds, min_tracking, cameras, algorithm, detector):
    """ Function to find the best parameters for the baselines and methodologies we use """
    idf1s = {}
    for dist_th in distance_thresholds:
        idf1s[dist_th] = []
        for cam in cameras:
            video_path = DATA_DIR / 'S03' / cam / 'vdo.avi'
            data_dir = DATA_DIR / 'S03' / cam
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

            metrics = IDF1Computation()
            y_pred = []

            moving_tracks = filter_moving_tracks(all_tracks, dist_th, min_tracking)
            detections = []
            for track in moving_tracks:
                detections.extend(track.tracking)
            detections = group_by_frame(detections)
            
            for frame_idx in range(0, get_video_length(str(video_path))):
                frame_detections = []
                for det in detections.get(frame_idx-1, []):
                    frame_detections.append(det)
                y_pred.append(frame_detections)
                metrics.add_frame_detections(y_true[frame_idx-1], y_pred[-1])
            summary = metrics.get_computation()
            idf1s[dist_th].append(summary['idf1']['metrics']*100)
            print(f'camera: {cam}, dist_th: {dist_th}, summary:\n{summary}')
    with open(str(RESULTS_DIR / f'idf1_seq3_{algorithm}_{detector}.pkl'), 'wb') as file:
        pickle.dump(idf1s, file)


def compute_results():
    distance_thresholds = [
        400, 500, 550, 600, 650, 700, 750, 800
    ]
    min_tracking = 5
    cameras = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    algorithm = 'tc'
    detector = 'ssd512'
    task_1_find_best_baseline(distance_thresholds, min_tracking, cameras, algorithm, detector)


def read_results(result_path, cameras=['c010', 'c011', 'c012', 'c013', 'c014', 'c015']):
    with open(result_path, 'rb') as file:
        results = pickle.load(file)
    for key, value in results.items():
        print(f'Threshold:{key}, Average: {np.mean(value)}')
        print('Per camera')
        for item, camera in zip(value, cameras):
            print(f'camera: {camera}, value:{item}')
        print('-'*10)


if __name__ == "__main__":
    compute_results()
    read_results(
        str(RESULTS_DIR / 'idf1_seq3_tc_ssd512.pkl'),
    )