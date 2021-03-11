import statistics

import random
import cv2
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image, ImageDraw
from pathlib import Path


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


def generate_frame_mious_plot(frames, mious, output_path):
    fig, ax = plt.subplots()
    ax.plot(frames, mious)

    ax.set(xlabel='frames', ylabel='mIoU', title='mIoU x Frame')
    fig.savefig(output_path)

    zeroDatapoint = 1
    minDatapoint = 1
    targetframe = 0
    targetZeroFrame = 0

    for datapoint, frame in zip(mious, frames):
        if datapoint < minDatapoint:
            if datapoint != 0:
                zeroDatapoint = datapoint
                targetZeroFrame = frame
            else:
                minDatapoint = datapoint
                targetframe = frame
    print(f'Generated plot into {output_path}')
    print(f"\nMin miou: {minDatapoint} at frame: {targetframe}")
    print(f"Zero miou: {zeroDatapoint} at frame: {targetZeroFrame}")
    print(f"Standard deviation: {statistics.stdev(mious)}")
    print(f"Mean: {statistics.mean(mious)}")
    print("_______________________________________________________")

def generate_noise_plot(x, y, xx, yy, label1, label2, xlabel, ylabel, title, output_path):
    plt.plot(x, y, label=label1)
    plt.plot(xx, yy, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f'Generating plot into {output_path}')

    
def generate_noise_plot_gif(x, y, xx, yy, label1, label2, xlabel, ylabel, title, output_path):
    plt.plot(x, y, label=label1)
    plt.plot(xx, yy, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f'Generating plot into {output_path}')