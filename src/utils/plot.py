import statistics

import random
import cv2
import numpy as np
import imageio

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, NoReturn


def visualize_random_frames(frames_directory, bounding_boxes, frames_to_show=4):
    random_frames = random.choices(list(bounding_boxes.items()), k=frames_to_show)
    for key, boxes in random_frames:
        image = frames_directory / f'output{str(key+1).zfill(3)}.jpg'
        with Image.open(image) as im:
            for box in boxes:
                draw = ImageDraw.Draw(im)
                draw.rectangle([(box.xtl, box.ytl), (box.xbr, box.ybr)], outline='green', width=6)
            display(im)

def plot_img(image, size=(10, 10), cmap=None, title='', save_root=None):
    plt.figure(figsize=size)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if save_root:
        plt.savefig(Path(f'{save_root}/{title.lower()}.png'))
    plt.show()
    plt.close()

def plot_roc(values, plot_range, xlabel, ylabel, title='', save_root=None):
    plt.plot(plot_range, values)
    plt.xticks(plot_range)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(Path(f'{save_root}/{title.lower()}.png'))
    plt.show()
    plt.close()

def plot_histogram_with_mean(histogram, mean, size=(10, 5), title='', cmap='viridis', label='', save_root=None):
    cm = plt.cm.get_cmap(cmap)
    plt.figure(figsize=size)
    plt.title(title)
    n, bins, patches = plt.hist(histogram, 25)
    col = (n-n.min())/(n.max()-n.min())
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    plt.axvline(mean, color='g', linestyle='dashed', linewidth=1, label=f'{label}:{round(mean, 2)}')
    plt.legend()
    plt.savefig(Path(f'{save_root}/{title.lower()}.png'))
    plt.show()
    plt.close()

def plot_optical_flow(gray_image, flow_image, sampling_step=10, size=(10, 10), title='', cmap='cool', save_root=None):
    flow_image = cv2.resize(flow_image, (0, 0), fx=1. / sampling_step, fy=1. / sampling_step)
    u = flow_image[:, :, 0]
    v = flow_image[:, :, 1]

    width = np.arange(0, flow_image.shape[1] * sampling_step, sampling_step)
    height = np.arange(0, flow_image.shape[0] * sampling_step, sampling_step)
    x, y = np.meshgrid(width, height)
    max_vect_length = max(np.max(u), np.max(v))

    plt.figure(figsize=size)
    plt.quiver(x, y, u, -v, np.hypot(u, v), scale=max_vect_length * sampling_step, cmap=cmap)
    plt.imshow(gray_image, alpha=0.8, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(Path(f'{save_root}/{title.lower()}.png'))
    plt.show()
    plt.close()

def plot_opt_flow_hsv(flow, scale=4):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    ang = np.arctan2(fy, fx) + np.pi
    mag = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = np.minimum(mag * scale, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb