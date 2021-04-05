import cv2
import numpy as np
import os
import time

from pathlib import Path
from src.block_matching import block_matching
from src.metrics.optical_flow import evaluate_flow

def task1_1():
    previous_frame = cv2.imread(str(Path.joinpath(Path(__file__).parent, './data/000045_10.png')), cv2.IMREAD_GRAYSCALE)
    current_frame = cv2.imread(str(Path.joinpath(Path(__file__).parent, './data/000045_11.png')), cv2.IMREAD_GRAYSCALE)
    flow_noc = cv2.imread(str(Path.joinpath(Path(__file__).parent, './data/gt_000045_10.png')))

    # Grid search
    algorithms = ['es', 'tss']
    metric_methods = ['euclidean', 'sad', 'ssd', 'mse', 'mad']
    block_sizes = [8, 16, 32]
    search_areas = [8*3, 16*3, 32*3]
    motion_type = [True, False]

    os.makedirs(str(Path.joinpath(Path(__file__).parent, './results/week4')), exist_ok=True)
    with open(str(Path.joinpath(Path(__file__).parent, './results/week4/task1_1.txt')), 'w') as f:
        for motion_type in [True, False]:
            for algorithm in algorithms:
                for metric_method in metric_methods:
                    for block_size in block_sizes:
                        for search_area in search_areas:
                            start = time.time()
                            flow = block_matching(current_frame=current_frame,
                                                  previous_frame=previous_frame,
                                                  forward=motion_type,
                                                  block_size=block_size,
                                                  search_area=search_area,
                                                  distance_method=metric_method,
                                                  algorithm=algorithm)
                            end = time.time()
                            metrics = evaluate_flow(flow_noc, flow)
                            f.write(f'Motion type: {motion_type} - Algorithm: {algorithm} - Metric: {metric_method} Block size: {block_size} - Search area: {search_area} - Time: {end-start} - Msen {metrics[0]} - Pepn {metrics[1]}')

if __name__ == "__main__":
    task1_1()