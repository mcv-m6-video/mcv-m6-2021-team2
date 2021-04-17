from pathlib import Path
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.readers.ai_city_reader import resolve_tracks_from_detections, group_by_frame
from src.video import get_frames_from_video, generate_video
from src.metrics.mot_metrics import IDF1Computation
from src.tracking import MaxOverlapTracker, filter_moving_tracks
from src.test_bm import block_matching_flow
from src.detection import Detection
from src.track import Track
from src.sort import Sort

import numpy as np
import cv2
import pickle

# -- CONSTANTS -- #
DATA_DIR = Path('data/AICity_track_data/train')
RESULTS_DIR = Path('results/week5/')

DETECTOR = 'mask_rcnn'
MIN_TRACKING = 5
DIST_THRESHOLD = 600
FLOW_METHOD = 'block_matching'

# Block matching parameters
forward = True
block_size = 32
search_area = 32
algorithm = 'tss'
distance = 'mse'

# Kalman Filter parameters
max_age = 1
min_hits = 3
iou_threshold = 0.3


def task_1_max_overlap(sequences, cameras):
    idf1s = {}
    for seq in sequences:
        idf1s[seq] = []
        for cam in cameras[seq]:
            video_path = DATA_DIR / seq / cam / 'vdo.avi'
            data_dir = DATA_DIR / seq / cam
            cap = cv2.VideoCapture(str(video_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_percentage = 1
            start = 0
            end = int(n_frames * video_percentage)

            # Groundtruth
            gt_reader = AICityChallengeAnnotationReader(str(data_dir / 'gt' / 'gt.txt'))
            gt = gt_reader.get_annotations(classes=['car'])

            # Detections
            det_reader = AICityChallengeAnnotationReader(str(data_dir / 'det' / f'det_{DETECTOR}.txt'))
            dets = det_reader.get_annotations(classes=['car'])
            tracker = MaxOverlapTracker()
            metrics = IDF1Computation()
            y_true = []
            y_pred = []
            all_tracks = []
            for frame_idx, current_frame in get_frames_from_video(str(video_path), start_frame=start, end_frame=end):
                current_detections = dets.get(frame_idx-1, [])
                all_tracks, tracks_on_frame = tracker.track_by_max_overlap(all_tracks, current_detections, optical_flow=None)
                y_true.append(gt.get(frame_idx-1, []))

            moving_tracks = filter_moving_tracks(all_tracks, DIST_THRESHOLD, MIN_TRACKING)
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
            idf1s[seq].append(summary['idf1']['metrics']*100)
            print(f'sequence: {seq}, camera: {cam}, dist_th: {DIST_THRESHOLD}, summary:\n{summary}')
    with open(str(RESULTS_DIR / f'max_overlap_{DETECTOR}.pkl'), 'wb') as file:
        pickle.dump(idf1s, file)


def task_1_kalman_filter(sequences, cameras):
    idf1s = {}
    for seq in sequences:
        idf1s[seq] = []
        for cam in cameras[seq]:
            video_path = DATA_DIR / seq / cam / 'vdo.avi'
            data_dir = DATA_DIR / seq / cam
            cap = cv2.VideoCapture(str(video_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_percentage = 1
            start = 0
            end = int(n_frames * video_percentage)

            # Groundtruth
            gt_reader = AICityChallengeAnnotationReader(str(data_dir / 'gt' / 'gt.txt'))
            gt = gt_reader.get_annotations(classes=['car'])

            # Detections
            det_reader = AICityChallengeAnnotationReader(str(data_dir / 'det' / f'det_{DETECTOR}.txt'))
            dets = det_reader.get_annotations(classes=['car'])
            tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
            metrics = IDF1Computation()
            y_true = []
            y_pred = []
            all_tracks = []
            for frame_idx, current_frame in get_frames_from_video(str(video_path), start_frame=start, end_frame=end):
                current_detections = dets.get(frame_idx-1, [])
                new_detections = tracker.update(np.array([[*detection.bbox, detection.score] for detection in current_detections]))
                new_detections = [Detection(frame_idx-1, int(detection[-1]), 'car', *detection[:4]) for detection in new_detections]
                all_tracks, tracks_on_frame = tracker.track_by_max_overlap(all_tracks, new_detections, optical_flow=None)
                y_true.append(gt.get(frame_idx-1, []))

            moving_tracks = filter_moving_tracks(all_tracks, DIST_THRESHOLD, MIN_TRACKING)
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
            idf1s[seq].append(summary['idf1']['metrics']*100)
            print(f'sequence: {seq}, camera: {cam}, dist_th: {DIST_THRESHOLD}, summary:\n{summary}')
    with open(str(RESULTS_DIR / f'kalman_{DETECTOR}.pkl'), 'wb') as file:
        pickle.dump(idf1s, file)


def task_1_max_overlap_with_flow(sequences, cameras):
    idf1s = {}
    for seq in sequences:
        idf1s[seq] = []
        for cam in cameras[seq]:
            video_path = DATA_DIR / seq / cam / 'vdo.avi'
            data_dir = DATA_DIR / seq / cam
            cap = cv2.VideoCapture(str(video_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_percentage = 1
            start = 0
            end = int(n_frames * video_percentage)

            # Groundtruth
            gt_reader = AICityChallengeAnnotationReader(str(data_dir / 'gt' / 'gt.txt'))
            gt = gt_reader.get_annotations(classes=['car'])

            # Detections
            det_reader = AICityChallengeAnnotationReader(str(data_dir / 'det' / f'det_{DETECTOR}.txt'))
            dets = det_reader.get_annotations(classes=['car'])
            tracker = MaxOverlapTracker()
            metrics = IDF1Computation()
            y_true = []
            y_pred = []
            all_tracks = []
            previous_frame = None
            for frame_idx, current_frame in get_frames_from_video(str(video_path), start_frame=start, end_frame=end):
                if previous_frame is None:
                    optical_flow = None
                else:
                    if FLOW_METHOD == 'farneback':
                        img_0 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                        img_1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                        optical_flow = cv2.calcOpticalFlowFarneback(img_0, img_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    elif FLOW_METHOD == 'block_matching':
                        optical_flow = block_matching_flow(
                            img_prev=previous_frame, img_next=current_frame, motion_type=forward,
                            block_size=block_size, search_area=search_area,
                            algorithm=algorithm, metric=distance
                        )
                    else:
                        raise ValueError(f'This {FLOW_METHOD} is not available')
                previous_frame = current_frame.copy()
                current_detections = dets.get(frame_idx-1, [])
                all_tracks, tracks_on_frame = tracker.track_by_max_overlap(all_tracks, current_detections, optical_flow=optical_flow)
                y_true.append(gt.get(frame_idx-1, []))

            moving_tracks = filter_moving_tracks(all_tracks, DIST_THRESHOLD, MIN_TRACKING)
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
            idf1s[seq].append(summary['idf1']['metrics']*100)
            print(f'sequence: {seq}, camera: {cam}, dist_th: {DIST_THRESHOLD}, summary:\n{summary}')
    with open(str(RESULTS_DIR / f'optical_flow_{DETECTOR}.pkl'), 'wb') as file:
        pickle.dump(idf1s, file)


def read_results(result_path, cameras):
    with open(result_path, 'rb') as file:
        results = pickle.load(file)
    for key, value in results.items():
        print(f'Threshold:{key}, Average: {np.mean(value)}')
        print('Per camera')
        for item, camera in zip(value, cameras[key]):
            print(f'camera: {camera}, value:{item}')
        print('-'*10)


if __name__ == "__main__":
    sequences = ['S01', 'S03', 'S04']
    cameras = {
        'S01': ['c001', 'c002', 'c003', 'c004', 'c005'],
        'S03': ['c010', 'c011', 'c012', 'c013', 'c014', 'c015'],
        's04': ['c016', 'c017', 'c018', 'c019', 'c020', 'c021', 'c022', 'c023', 'c024',
                'c025', 'c026', 'c027', 'c028', 'c029', 'c030', 'c031', 'c032', 'c033', 'c034',
                'c035', 'c036', 'c037', 'c038', 'c039', 'c040'
        ]
    }
    task_1_max_overlap(sequences, cameras)
    read_results(
        str(RESULTS_DIR / 'max_overlap_mask_rcnn.pkl'),
        cameras
    )