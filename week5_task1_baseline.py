from pathlib import Path
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.readers.ai_city_reader import resolve_tracks_from_detections, group_by_frame
from src.video import get_frames_from_video, generate_video, get_video_length
from src.metrics.mot_metrics import IDF1Computation
from src.tracking import filter_moving_tracks

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
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
            
            for frame_idx in trange(0, get_video_length(str(video_path))):
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
        400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800
    ]
    min_tracking = 5
    cameras = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    algorithm = ['deepsort', 'moana', 'tc']
    detector = ['mask_rcnn', 'yolo3', 'ssd512']
    for alg in algorithm:
        for det in detector:
            task_1_find_best_baseline(distance_thresholds, min_tracking, cameras, alg, det)


def show_all_results():
    algorithms = ['deepsort', 'moana', 'tc']
    detectors = ['mask_rcnn', 'yolo3', 'ssd512']
    read_results(algorithms, detectors)


def read_results(algorithms, detectors, cameras=['c010', 'c011', 'c012', 'c013', 'c014', 'c015'], per_camera=False):
    for algorithm in algorithms:
        plt.figure(figsize=(8,5))
        for detector in detectors:
            result_path = str(RESULTS_DIR / f'idf1_seq3_{algorithm}_{detector}.pkl')
            with open(result_path, 'rb') as file:
                results = pickle.load(file)
            print(f'Results for algorithm: {algorithm} and detector: {detector}')
            thresholds = list(results.keys())
            averages = []
            for key, value in results.items():
                print(f'Threshold:{key}, Average: {np.mean(value)}')
                averages.append(np.mean(value))
                if per_camera:
                    print('Per camera:')
                    for item, camera in zip(value, cameras):
                        print(f'camera: {camera}, value:{item}')
                    print('-'*10)
            plt.plot(thresholds, averages)
            print('#'*10)
        title = f'Results for {algorithm}'
        plt.title(title)
        plt.xlabel('Distance between Centroids')
        plt.ylabel('IDF1')
        plt.legend(detectors)
        plt.savefig(Path(f'{RESULTS_DIR}/{title.lower()}.png'))
        plt.show()


if __name__ == "__main__":
    #compute_results()
    show_all_results()