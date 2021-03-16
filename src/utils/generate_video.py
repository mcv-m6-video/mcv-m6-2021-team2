from typing import List
from src.annotation import Annotation
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
import cv2


from typing import List
from src.annotation import Annotation
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
import cv2

def generate_video(video_path: str,
                   frames_index: List[int],
                   gt_annons: List[Annotation],
                   predict_annons: List[Annotation],
                   title: str,
                   output_path: str,
                   frame_miou: List[float] = None):

    cap = cv2.VideoCapture(video_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if frame_miou:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        image = ax[0].imshow(np.zeros((height, width)))
        line, = ax[1].plot(frames_index, frame_miou)
        artists = [image, line]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        image = ax.imshow(np.zeros((height, width)))
        artists = [image]

    def update(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_index[i])

        ret, img = cap.read()
        for d in [gt for gt in gt_annons if gt.frame == frames_index[i]]:
            cv2.rectangle(img, (int(d.left), int(d.top)), (int(d.width), int(d.height)), (0, 255, 0), 2)
        for d in [predict for predict in predict_annons if predict.frame == frames_index[i]]:
            cv2.rectangle(img, (int(d.left), int(d.top)), (int(d.width), int(d.height)), (0, 0, 255), 2)

        artists[0].set_data(img[:, :, ::-1])

        if frame_miou:
            artists[1].set_data(frames_index[:i + 1], frame_miou[:i + 1])

        return artists

    ani = FuncAnimation(fig, update, len(frames_index), interval=2, blit=True)

    if frame_miou:
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].legend(handles=[patches.Patch(color="green", label="GT"), patches.Patch(color="red", label="Pred")])
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel('#frame')
        ax[1].set_ylabel('mean IoU')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(handles=[patches.Patch(color="green", label="GT"), patches.Patch(color="red", label="Pred")])

    fig.suptitle(title)
    ani.save(output_path, writer='imagemagick')

    cap.release()
    print(f'Gif generated with name {output_path}')