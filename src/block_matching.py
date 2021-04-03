import cv2
from tqdm import tqdm
from typing import Tuple, List
import numpy as np

from src.metrics.distance import calc_distance
from pathlib import Path


def block_matching(current_frame: np.array,
                   previous_frame: np.array,
                   forward: bool,
                   block_size: int,
                   search_area: int,
                   distance_method: str = 'euclidean') -> np.array:

    if forward:
        source = previous_frame
        target = current_frame
    else:
        source = current_frame
        target = previous_frame

    source_height, source_width = source.shape

    result = np.zeros((source_height, source_width, 2))

    for y_source in tqdm(range(0, source_height-block_size, block_size)):
        for x_source in range(0, source_width-block_size, block_size):
            source_block = source[y_source:y_source+block_size, x_source:x_source+block_size]

            xtl = max(0, x_source-search_area)
            ytl = max(0, y_source-search_area)
            xbr = min(source_width, x_source+block_size+search_area)
            ybr = min(source_height, y_source+block_size+search_area)

            target_area = target[ytl:ybr, xtl:xbr]
            target_height, target_width = target_area.shape

            min_dist = np.inf
            u = 0
            v = 0
            for y_target in range(0, target_height-block_size, 1):
                for x_target in range(0, target_width-block_size, 1):
                    target_block = target_area[y_target:y_target+block_size, x_target:x_target+block_size]

                    dist = calc_distance(source_block, target_block, distance_method)
                    if dist < min_dist:
                        min_dist = dist
                        v = y_target - (y_source - ytl)
                        u = x_target - (x_source - xtl)

            result[y_source:y_source+block_size, x_source:x_source+block_size] = [u, v]

    return result