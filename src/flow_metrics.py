import numpy as np
import cv2

def calc_optical_flow(gt: np.array, pred: np.array):
    discard = gt[:, :, -1] != 0

    ch0 = gt[:, :, 0] - pred[:, :, 0]
    ch1 = gt[:, :, 1] - pred[:, :, 1]

    mse = np.sqrt(ch0**2 + ch1**2)
    error = mse[discard]

    msen = np.mean(error)
    pepn = np.sum(error > 3) / len(error)

    return mse, error, msen, pepn * 100


def magnitude_flow(flow_image, dilate=True):
    if len(flow_image.shape) > 2:
        magnitude, angle = cv2.cartToPolar(flow_image[:, :, 0], flow_image[:, :, 1])
        flow_image = magnitude

    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        flow_image = cv2.dilate(flow_image, kernel, iterations=1)
    
    return flow_image