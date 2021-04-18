import time

import numpy as np
import cv2
from tqdm import trange


def distance(x1: np.ndarray, x2: np.ndarray, metric='euclidean'):
    if metric == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif metric == 'sad':
        return np.sum(np.abs(x1 - x2))
    elif metric == 'mad':
        return np.mean(np.abs(x1 - x2))
    elif metric == 'ssd':
        return np.sum((x1 - x2) ** 2)
    elif metric == 'mse':
        return np.mean((x1 - x2) ** 2)
    else:
        raise ValueError(f'Unknown distance metric: {metric}')


def block_matching(reference, target, metric='euclidean', algorithm='es'):
    """
    Search reference in target and return the position with maximum correlation
    """
    if algorithm == 'es':
        # exhaustive search
        min_dist = np.inf
        pos = (0, 0)
        for i in range(target.shape[0]-reference.shape[0]):
            for j in range(target.shape[1]-reference.shape[1]):
                dist = distance(reference, target[i:i+reference.shape[0], j:j+reference.shape[1]], metric)
                if dist < min_dist:
                    pos = (i, j)
                    min_dist = dist
        return pos
    elif algorithm == 'tss':
        # three step search
        step = 8
        orig = ((target.shape[0]-reference.shape[0])//2, (target.shape[1]-reference.shape[1])//2)
        step = (min(step, orig[0]), min(step, orig[1]))
        while step[0] > 1 and step[1] > 1:
            min_dist = np.inf
            pos = orig
            for i in [orig[0]-step[0], orig[0], orig[0]+step[0]]:
                for j in [orig[1]-step[1], orig[1], orig[1]+step[1]]:
                    dist = distance(reference, target[i:i + reference.shape[0], j:j + reference.shape[1]], metric)
                    if dist < min_dist:
                        pos = (i, j)
                        min_dist = dist
            orig = pos
            step = (step[0]//2, step[1]//2)
        return orig
    else:
        raise ValueError(f'Unknown block matching algorithm: {algorithm}')


def block_matching_flow(img_prev: np.ndarray, img_next: np.ndarray, block_size=16, search_area=16,
                        motion_type=True, metric='euclidean', algorithm='es'):
    """
    Compute block-matching based motion estimation
    """
    if motion_type:
        reference = img_prev
        target = img_next
    else:
        reference = img_next
        target = img_prev

    assert (reference.shape == target.shape)
    height, width = reference.shape[:2]

    flow_field = np.zeros((height, width, 2), dtype=float)
    for i in trange(0, height - block_size, block_size):
        for j in range(0, width - block_size, block_size):
            ii = max(i-search_area, 0)
            jj = max(j-search_area, 0)
            disp = block_matching(reference[i:i+block_size, j:j+block_size],
                                  target[ii: min(i+block_size+search_area, height),
                                         jj: min(j+block_size+search_area, width)],
                                  metric, algorithm)
            u = disp[1] - (j - jj)
            v = disp[0] - (i - ii)
            flow_field[i:i + block_size, j:j + block_size, :] = [u, v]

    return flow_field
