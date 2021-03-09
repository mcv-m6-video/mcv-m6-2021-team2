import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def plot_arrows(gray_image, flow_image, sampling_step, frame_id, name):
    flow_image = cv2.resize(flow_image, (0, 0), fx=1. / sampling_step, fy=1. / sampling_step)
    u = flow_image[:, :, 0]
    v = flow_image[:, :, 1]

    width = np.arange(0, flow_image.shape[1] * sampling_step, sampling_step)
    height = np.arange(0, flow_image.shape[0] * sampling_step, sampling_step)
    x, y = np.meshgrid(width, height)
    max_vect_length = max(np.max(u), np.max(v))

    plt.figure()
    plt.quiver(x, y, u, -v, np.hypot(u, v), scale=max_vect_length * sampling_step, cmap='cool')
    plt.imshow(gray_image, alpha=0.8, cmap='gray')
    plt.title(f'Flow Results {name}-{frame_id}')
    plt.axis('off')
    plt.savefig(os.path.join('results/week1', f'flow_results_{name}-{frame_id}.png'))
    plt.close()
