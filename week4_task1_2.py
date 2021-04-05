import cv2
import time
import numpy as np

from pathlib import Path
from src.pyflow import pyflow
from src.hornschunk import HornSchunck
from src.lk_flow import LK_flow
from src.utils.flow_reader import read_flow_img
from src.utils.plot import plot_optical_flow, plot_opt_flow_hsv, plot_img
from src.metrics.optical_flow import evaluate_flow, compute_msen, compute_pepn

RESULTS_DIR = Path('results/week4')
DATA_PATH = Path('data/optical_flows')


def task_1_2(algorithm='pyflow'):
    # Read images
    img_0 = cv2.imread(str(DATA_PATH / '000045_10.png'), cv2.IMREAD_GRAYSCALE)
    img_1 = cv2.imread(str(DATA_PATH / '000045_11.png'), cv2.IMREAD_GRAYSCALE)
    flow_gt = read_flow_img(str(DATA_PATH / 'flows/000045_10.png'))

    # Compute Off the Shelf optical flow depending on the algorithm
    if algorithm == 'pyflow':
        im0 = np.atleast_3d(img_0.astype(float) / 255.)
        im1 = np.atleast_3d(img_1.astype(float) / 255.)

        tic = time.time()
        u, v, im2W = pyflow.coarse2fine_flow(
            im0, im1, alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7,
            nInnerFPIterations=1, nSORIterations=30, colType=1
        )
        toc = time.time()
        flow = np.dstack((u, v))
    elif algorithm == 'hornschunck':
        tic = time.time()
        u, v = HornSchunck(img_0, img_1, alpha=0.012, Niter=8)
        toc = time.time()
        flow = np.dstack((u, v))
    elif algorithm == 'lk':
        flow, tic, toc = LK_flow(img_0, img_1)
    elif algorithm == 'farneback':
        tic = time.time()
        flow = cv2.calcOpticalFlowFarneback(img_0, img_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        toc = time.time()
    else:
        raise ValueError(f'The Algorithm {algorithm} is not available')

    msen, pepn = evaluate_flow(flow_gt, flow)
    print(f'MSEN: {msen:.4f}, PEPN: {pepn:.4f}, runtime: {toc-tic:.3f}s')
    print('Saving Results...')
    hsv_gt = plot_opt_flow_hsv(flow_gt)
    hsv = plot_opt_flow_hsv(flow)
    plot_img(hsv, cmap=None, title=f'HSV_Representation_{algorithm}', save_root=RESULTS_DIR)
    plot_img(hsv_gt, cmap=None, title=f'HSV_GT_Representation', save_root=RESULTS_DIR)
    print('HSV saved')
    plot_optical_flow(img_0, flow, title=f'Optical_Flow_using_{algorithm}', save_root=RESULTS_DIR)
    plot_optical_flow(img_0, flow_gt, title=f'GT_Optical_Flow', save_root=RESULTS_DIR)
    print('Done.')

if __name__ == "__main__":
    task_1_2(algorithm='lk')