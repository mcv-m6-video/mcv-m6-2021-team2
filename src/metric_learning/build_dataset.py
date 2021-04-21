from typing import List, NoReturn
import os
from pathlib import Path
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from src.video import get_frames_from_video
from src.detection import Detection
from src.readers.ai_city_reader import AICityChallengeAnnotationReader


def plot_random_crops(img_size: int = 20,
                      num_rows: int = 4,
                      num_colums: int = 5,
                      img_name: str = 'ml_sample.png') -> NoReturn:
    ml_database_path = Path.joinpath(Path(__file__).parent, '../../data/ml_database/train')
    if not ml_database_path.exists():
        raise ValueError('The database for metric learning is not created. Create it first.')

    fig = plt.figure(figsize=(img_size, img_size))
    image_grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_colums), axes_pad=0.1)

    for ax, id_folder in zip(image_grid, list(ml_database_path.iterdir())[:20]):
        crop_path = str(list(id_folder.iterdir())[0])
        img = cv2.imread(crop_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')

    plt.axis('off')
    plt.savefig(img_name)
    plt.close()


def crop_car_from_frame(detection: Detection,
                            frame: np.array,
                            img_size: int) -> np.array:
    car = frame[int(detection.ytl):int(detection.ybr),int(detection.xtl):int(detection.xbr)]
    return cv2.resize(car, (img_size, img_size))


def generate_sequence(seq: str,
                      dataset_path: str,
                      output_path: str,
                      img_size: int) -> NoReturn:
    for cam in Path(Path.joinpath(Path(dataset_path), seq)).iterdir():
        if cam.is_dir():
            output_dataset_path = str(Path.joinpath(Path(__file__).parent, f'../../data/ml_database/{seq}/{cam.name}'))

            video_path = str(Path.joinpath(cam, f'./vdo.avi'))

            gt_path = str(Path.joinpath(cam, f'./gt/gt.txt'))
            gt_annotations = AICityChallengeAnnotationReader(gt_path).get_annotations(classes=['car'])

            print(f'Seq {seq} - Cam {cam.name}')
            for frame_idx, frame in get_frames_from_video(video_path):
                if (frame_idx - 1) not in gt_annotations:
                    continue

                for det in gt_annotations[frame_idx - 1]:
                    if det.width >= img_size and det.height >= img_size:
                        crop = crop_car_from_frame(det, frame, img_size)

                        output_crop_path = str(Path.joinpath(Path(output_path), f'{det.id}/{cam.name}_{frame_idx - 1}.png'))
                        os.makedirs(str(Path.joinpath(Path(output_path), str(det.id))), exist_ok=True)

                        cv2.imwrite(output_crop_path, crop)

def generate_metric_learning_database(dataset_path: str,
                                      train_seq: List[str],
                                      test_seq: List[str],
                                      img_size: int) -> NoReturn:

    if Path.joinpath(Path(__file__).parent, f'../../data/ml_database').exists():
        shutil.rmtree(str(Path.joinpath(Path(__file__).parent, f'../../data/ml_database')))

    for seq in train_seq:
        output_path = str(Path.joinpath(Path(__file__).parent, '../../data/ml_database/train'))
        generate_sequence(seq=seq,
                          dataset_path=dataset_path,
                          output_path=output_path,
                          img_size=img_size)
    for seq in test_seq:
        output_path = str(Path.joinpath(Path(__file__).parent, '../../data/ml_database/test'))
        generate_sequence(seq=seq,
                          dataset_path=dataset_path,
                          output_path=output_path,
                          img_size=img_size)


