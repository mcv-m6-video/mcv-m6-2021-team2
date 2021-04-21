from pathlib import Path
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.readers.ai_city_reader import resolve_tracks_from_detections, group_by_frame
from src.video import get_frames_from_video, generate_video, get_video_length
from src.metrics.mot_metrics import IDF1Computation
from src.tracking import MaxOverlapTracker, filter_moving_tracks
from src.block_matching import block_matching_flow
from src.detection import Detection
from src.track import Track
from src.sort import Sort

from tqdm import tqdm, trange
import numpy as np
import cv2
import pickle

# -- CONSTANTS -- #
DATA_DIR = Path('data/AICity_track_data/train')
RESULTS_DIR = Path('results/week5/')

DETECTOR = 'mask_rcnn'
MIN_TRACKING = 5
DIST_THRESHOLD = 725
FLOW_METHOD = 'farneback'


def video_max_overlap(seq, cam):
    writer = imageio.get_writer(str(RESULTS_DIR / f'video.gif'), fps=10)
    video_path = DATA_DIR / seq / cam / 'vdo.avi'
    data_dir = DATA_DIR / seq / cam

    # Groundtruth
    gt_reader = AICityChallengeAnnotationReader(str(data_dir / 'gt' / 'gt.txt'))
    gt = gt_reader.get_annotations(classes=['car'])

    # Detections
    det_reader = AICityChallengeAnnotationReader(str(data_dir / 'det' / f'det_{detector}.txt'))
    dets = det_reader.get_annotations(classes=['car'])
    tracker = MaxOverlapTracker()
    all_tracks = []

    for frame_idx in trange(0, get_video_length(str(video_path))):
        current_detections = dets.get(frame_idx-1, [])
        all_tracks, tracks_on_frame = tracker.track_by_max_overlap(all_tracks, current_detections, optical_flow=None)
        for track in tracks_on_frame:
            detection = track.previous_detection
            cv2.rectangle(frame, (int(detection.xtl), int(detection.ytl)), (int(detection.xbr), int(detection.ybr)), track.color, 2)
            cv2.rectangle(frame, (int(detection.xtl), int(detection.ytl)), (int(detection.xbr), int(detection.ytl) - 15), track.color, -2)
            cv2.putText(frame, str(detection.id), (int(detection.xtl), int(detection.ytl)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            for tt in track.tracking:
                cv2.circle(frame, tt.center, 5, track.color, -1)

    moving_tracks = filter_moving_tracks(all_tracks, DIST_THRESHOLD, MIN_TRACKING)
    detections = []
    for track in moving_tracks:
        detections.extend(track.tracking)
    detections = group_by_frame(detections)

    for frame_idx in trange(0, get_video_length(str(video_path))):
        frame_detections = []
        for det in detections.get(frame_idx-1, []):
            frame_detections.append(det)