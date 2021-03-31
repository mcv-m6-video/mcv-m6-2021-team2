import cv2
import time
import numpy as np

from pathlib import Path
from src.pyflow import pyflow
from src.hornschunk import HornSchunck
from src.utils.flow_reader import read_flow_img
from src.utils.plot import plot_optical_flow, plot_opt_flow_hsv, plot_img
from src.metrics.optical_flow import evaluate_flow

RESULTS_DIR = Path('Results/week4')
DATA_PATH = Path('data/optical_flow')


def task_1_2(algorithm='pyflow'):
    # Read images
    img_0 = cv2.imread(str(DATA_PATH / '000045_10.png'), cv2.IMREAD_GRAYSCALE)
    img_1 = cv2.imread(str(DATA_PATH / '000045_11.png'), cv2.IMREAD_GRAYSCALE)
    flow_gt = read_flow_img(str(DATA_PATH / 'flows/000045_10.png'))

    # Compute Off the Shelf optical flow depending on the algorithm
    if algorithm == 'pyflow':
        im0 = np.atleast_3d(img_0.astype(float) / 255.)
        im1 = np.atleast_3d(img_1.astype(float) / 255.)

        # flow options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

        tic = time.time()
        u, v, im2W = pyflow.coarse2fine_flow(
            im0, im1, alpha, ratio, minWidth, nOuterFPIterations,
            nInnerFPIterations, nSORIterations, colType
        )
        toc = time.time()
        flow = np.dstack((u, v))
    elif algorithm == 'hornschunck':
        alpha = 0.012
        iterations = 8
        tic = time.time()
        u, v = HornSchunck(img_0, img_1, alpha=alpha, Niter=iterations)
        toc = time.time()
        flow = np.dstack((u, v))
    else:
        raise ValueError(f'The Algorithm {algorithm} is not available')

    msen, pepn = evaluate_flow(flow_gt, flow)
    print(f'MSEN: {msen:.4f}, PEPN: {pepn:.4f}, runtime: {toc-tic:.3f}s')
    hsv = plot_opt_flow_hsv(flow)
    plot_img(hsv)
    plot_optical_flow(img_0, flow, title=f'Optical Flow using {algorithm}', save_root=RESULTS_DIR)

if __name__ == "__main__":
    task_1_2(algorithm='pyflow')