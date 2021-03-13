import statistics

import random
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

def generate_video(video_path, frames, frame_ious, gt_bb, dd_bb, title='', save_root=None, size=(10,5)):
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fig, ax = plt.subplots(2, 1, figsize=size)
    image = ax[0].imshow(np.zeros((height, width)))
    line, = ax[1].plot(frames, frame_ious)
    artists = [image, line]

    def update(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, img = cap.read()
        for bb in gt_bb[frames[i]]:
            cv2.rectangle(img, (int(bb.xtl), int(bb.ytl)), (int(bb.xbr), int(bb.ybr)), (0, 255, 0), 4)
        for bb in dd_bb[frames[i]]:
            cv2.rectangle(img, (int(bb.xtl), int(bb.ytl)), (int(bb.xbr), int(bb.ybr)), (0, 0, 255), 4)
        artists[0].set_data(img[:, :, ::-1])
        artists[1].set_data(frames[:i + 1], frame_ious[:i + 1])
        return artists

    ani = animation.FuncAnimation(fig, update, len(frames), interval=2, blit=True)

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('#frame')
    ax[1].set_ylabel('mean IoU')
    fig.suptitle(title)
    ani.save(Path(f'{save_root}/{title.lower()}.gif', writer='imagemagick'))
    plt.show()
    plt.close()