import numpy as np
import cv2


def compute_msen(flow_gt: np.ndarray, flow_pred: np.ndarray, th: int = 3):
    """Mean Square Error in Non-occluded areas"""

    # compute mse, filtering discarded vectors
    u_diff = flow_gt[:, :, 0] - flow_pred[:, :, 0]
    v_diff = flow_gt[:, :, 1] - flow_pred[:, :, 1]
    squared_error = np.sqrt(u_diff**2 + v_diff**2)

    # discard vectors which from occluded areas (occluded = 0)
    non_occluded_idx = flow_gt[:, :, 2] != 0
    err_non_occ = squared_error[non_occluded_idx]

    msen = np.mean(err_non_occ)

    return squared_error, err_non_occ, msen


def compute_pepn(err: np.ndarray, n_pixels: int, th: int) -> float:
    """Percentage of Erroneous Pixels in Non-occluded areas"""
    return (np.sum(err > th) / n_pixels) * 100


def evaluate_flow(flow_gt, flow):
    err = np.sqrt(np.sum((flow_gt[..., :2] - flow) ** 2, axis=2))
    noc = flow_gt[..., 2].astype(bool)
    msen = np.mean(err[noc] ** 2)
    pepn = np.sum(err[noc] > 3) / err[noc].size
    return msen, pepn


def compute_magnitude(flow_image, dilate=True):
    if len(flow_image.shape) > 2:
        magnitude, angle = cv2.cartToPolar(flow_image[:, :, 0], flow_image[:, :, 1])
        flow_image = magnitude
    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        flow_image = cv2.dilate(flow_image, kernel, iterations=1)
    return flow_image
