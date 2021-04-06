import cv2
import numpy as np
import os
import time

from pathlib import Path
from src.block_matching import block_matching
from src.metrics.optical_flow import evaluate_flow
from src.utils.flow_reader import read_flow_img
from src.test_bm import block_matching_flow

def task1_1():
    previous_frame = cv2.imread(str(Path.joinpath(Path(__file__).parent, './data/000045_10.png')), cv2.IMREAD_GRAYSCALE)
    current_frame = cv2.imread(str(Path.joinpath(Path(__file__).parent, './data/000045_11.png')), cv2.IMREAD_GRAYSCALE)
    flow_noc = read_flow_img(str(Path.joinpath(Path(__file__).parent, './data/gt_000045_10.png')))

    # Grid search
    algorithms = ['es']
    metric_methods = ['euclidean', 'sad', 'ssd', 'mse', 'mad']
    block_sizes = [8, 16, 32]
    search_areas = [32, 64, 128]

    os.makedirs(str(Path.joinpath(Path(__file__).parent, './results/week4')), exist_ok=True)
    with open(str(Path.joinpath(Path(__file__).parent, './results/week4/other_group_task1_1_es_backward.txt')), 'w') as f:
        for motion_type in [False]:
            for algorithm in algorithms:
                for metric_method in metric_methods:
                    for block_size in block_sizes:
                        for search_area in search_areas:
                            start = time.time()
                            """
                            flow = block_matching(current_frame=current_frame,
                                                  previous_frame=previous_frame,
                                                  forward=motion_type,
                                                  block_size=block_size,
                                                  search_area=search_area,
                                                  distance_method=metric_method,
                                                  algorithm=algorithm)
                            """
                            flow = block_matching_flow(previous_frame,
                                                       current_frame, block_size,
                                                       search_area, motion_type,
                                                       metric_method, algorithm)
                            end = time.time()
                            metrics = evaluate_flow(flow_noc, flow)
                            print(f'{motion_type},{algorithm},{metric_method},{block_size},{search_area},{end-start},{metrics[0]},{metrics[1]}\n')
                            f.write(f'{motion_type},{algorithm},{metric_method},{block_size},{search_area},{end-start},{metrics[0]},{metrics[1]}\n')

if __name__ == "__main__":
    task1_1()
    """
    with open(str(Path.joinpath(Path(__file__).parent, './results/week4/task1_1.txt')), 'r') as f:

        from operator import itemgetter
        import copy
        content = []
        for line in f.readlines():
            content.append(line.split(','))

        print('Best msen')
        print(*(sorted(content, key=lambda x: x[6], reverse=False)[:5]), sep="\n")
        print('Best time')
        print(*(sorted(content, key=lambda x: x[5], reverse=False)[:5]), sep="\n")
        print('Best pepn')
        print(*(sorted(content, key=lambda x: x[7], reverse=False)[:5]), sep="\n")
    """