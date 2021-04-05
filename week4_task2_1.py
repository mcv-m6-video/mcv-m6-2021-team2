import time
import imageio
import cv2
import numpy as np

from pathlib import Path
from tqdm import trange
from src.pyflow import pyflow
from src.block_matching import block_matching

RESULTS_DIR = Path('results/week4')
WIDTH = 600
HEIGHT = 350

FORWARD = True
BLOCK_SIZE = 8
SEARCH_AREA = 8*3

def task_2_1(algorithm='block_matching', method='average'):
    cap = cv2.VideoCapture('data/test_stab.mp4')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_video = imageio.get_writer(str(RESULTS_DIR / f'stabilization_{algorithm}_{method}.gif'), fps=fps)

    previous_frame = None
    for k in trange(0, n_frames, desc='Applying Stabilization'):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'The frame {k} could not be readed.')
        current_frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        if k == 0:
            frame_stabilized = current_frame
        else:
            flow = compute_optical_flow(algorithm, previous_frame, current_frame)
            frame_stabilized = apply_motion(current_frame, flow, method)
        previous_frame = current_frame

        out_video.append_data(frame_stabilized)


def compute_optical_flow(algorithm, previous_frame, current_frame):
    if algorithm == 'block_matching':
        flow = block_matching(current_frame, previous_frame, FORWARD, BLOCK_SIZE, SEARCH_AREA)
    elif algorithm == 'pyflow':
        u, v, im2W = pyflow.coarse2fine_flow(
            previous_frame, current_frame, alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7,
            nInnerFPIterations=1, nSORIterations=30, colType=1
        )
        flow = np.dstack((u, v))
    else:
        raise ValueError('This option is not available.')
    return flow


def apply_motion(current_frame, flow, momentum, method):
    if method == 'average':
        x_t = flow[:, :, 0].mean()  # x translation
        y_t = flow[:, :, 1].mean()  # y translation
    elif method == 'median':
        x_t = np.median(flow[:, :, 0])
        y_t = np.median(flow[:, :, 1])
    else:
        raise ValueError('This option is not available')
    H = np.array([
        [1, 0, -x_t],
        [0, 1, -y_t]
    ], dtype=np.float32)
    return cv2.warpAffine(current_frame, H, (WIDTH, HEIGHT))

if __name__ == "__main__":
    task_2_1(algorithm='block_matching')