import cv2
from tqdm import tqdm
from typing import Tuple, List
import numpy as np

from src.metrics.distance import calc_distance
from pathlib import Path


def es(block_size: int,
       target_area: np.array,
       source_block: np.array,
       distance_method: str = 'euclidean') -> Tuple[int, int]:

    target_height, target_width = target_area.shape
    min_dist = np.inf
    pos = (0, 0)

    for y_target in range(0, target_height-block_size, 1):
        for x_target in range(0, target_width-block_size, 1):
            target_block = target_area[y_target:y_target+block_size, x_target:x_target+block_size]

            dist = calc_distance(source_block, target_block, distance_method)
            if dist < min_dist:
                min_dist = dist
                pos = (y_target, x_target)

    return pos


def tss(block_size: int,
        search_area: np.array,
        source_block: np.array,
        distance_method: str = 'euclidean') -> Tuple[int, int]:

    def get_block(center: Tuple[int, int],
                  block_size: int,
                  source_block: np.array,
                  image: np.array) -> np.array:
        xtl = max(0, center[1]-block_size//2)
        ytl = max(0, center[0]-block_size//2)
        xbr = min(image.shape[1], center[1]+block_size//2)
        ybr = min(image.shape[0], center[0]+block_size//2)

        target_block = image[ytl:ybr, xtl:xbr]
        if target_block.shape != source_block.shape:
            return None
        return target_block

    step = 8
    origin = (search_area.shape[0]//2, search_area.shape[1]//2)

    min_dist = np.inf
    pos = origin

    while step != 1:
        point_list = [
            (origin[0]-step, origin[1]-step), (origin[0]+step, origin[1]+step), (origin[0]-step, origin[1]+step),
            (origin[0]+step, origin[1]-step), (origin[0]-step, origin[1]), (origin[0]+step, origin[1]),
            (origin[0], origin[1]+step), (origin[0], origin[1]+step), origin
        ]

        for center in point_list:
            target_block = get_block(center, block_size, source_block, search_area)

            if target_block is not None:
                dist = calc_distance(source_block, target_block, distance_method)

                if dist < min_dist:
                    min_dist = dist
                    pos = center

        origin = pos
        step = step//2

    return (origin[0]-block_size//2, origin[1]-block_size//2)

def ntss(block_size: int,
         search_area: np.array,
         source_block: np.array,
         distance_method: str = 'euclidean') -> Tuple[int, int]:

    def get_block(center: Tuple[int, int],
                  block_size: int,
                  source_block: np.array,
                  image: np.array) -> np.array:
        xtl = max(0, center[1]-block_size//2)
        ytl = max(0, center[0]-block_size//2)
        xbr = min(image.shape[1], center[1]+block_size//2)
        ybr = min(image.shape[0], center[0]+block_size//2)

        target_block = image[ytl:ybr, xtl:xbr]
        if target_block.shape != source_block.shape:
            return None
        return target_block

    step = 8
    origin = (search_area.shape[0]//2, search_area.shape[1]//2)

    min_dist = np.inf
    pos = origin

    while step != 1:
        point_list = [
            (origin[0]-step, origin[1]-step), (origin[0]+step, origin[1]+step), (origin[0]-step, origin[1]+step),
            (origin[0]+step, origin[1]-step), (origin[0]-step, origin[1]), (origin[0]+step, origin[1]),
            (origin[0], origin[1]+step), (origin[0], origin[1]+step), origin
        ]

        for center in point_list:
            target_block = get_block(center, block_size, source_block, search_area)

            if target_block is not None:
                dist = calc_distance(source_block, target_block, distance_method)

                if dist < min_dist:
                    min_dist = dist
                    pos = center

        if step == 8 and pos == origin: # step 1
            return (origin[0]-block_size//2, origin[1]-block_size//2)

        origin = pos
        step = step//2

    return (origin[0]-block_size//2, origin[1]-block_size//2)


def block_matching(current_frame: np.array,
                   previous_frame: np.array,
                   forward: bool,
                   block_size: int,
                   search_area: int,
                   distance_method: str = 'euclidean',
                   algorithm: str = 'tss') -> np.array:

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

            if algorithm == 'es':
                point = es(block_size, target_area, source_block)
            elif algorithm == 'tss':
                point = tss(block_size, target_area, source_block)
            else:
                raise NotImplementedError(f'The algorithm: {algorithm} is not implemented yet.')

            u = point[1] - (x_source - xtl)
            v = point[0] - (y_source - ytl)
            result[y_source:y_source+block_size, x_source:x_source+block_size] = [u, v]

    return result