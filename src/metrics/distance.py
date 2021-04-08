import numpy as np


def calc_distance(source: np.array, target: np.array, method: str) -> float:
    if method == 'euclidean':
        return euclidean(source, target)
    elif method == 'sad':
        return sad(source, target)
    elif method == 'ssd':
        return ssd(source, target)
    elif method == 'mse':
        return mse(source, target)
    elif method == 'mad':
        return mad(source, target)
    else:
        raise NotImplementedError(f'Method {method} is not implemented yet.')

def euclidean(source: np.array, target: np.array):
    return np.linalg.norm(source - target)

def sad(source: np.array, target: np.array):
    return np.sum(np.abs(source - target))

def ssd(source: np.array, target: np.array):
    return np.sum((source - target) ** 2)

def mse(source: np.array, target: np.array):
    return np.mean((source - target) ** 2)

def mad(source: np.array, target: np.array):
    return np.mean(np.abs(source - target))