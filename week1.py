from typing import List
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

from tools.annotation_reader import read_annotations
from src.metrics import IoU, mAP
from src.annotation import Annotation
from src.read_flow_img import read_flow_img
from src.flow_metrics import calc_optical_flow, magnitude_flow
from src.plot import plot_arrows
from tools.plot import generate_frame_mious_plot, generate_noise_plot
from tools.generate_video import generate_video

OUTPUT_FOLDER = "outputs"

def task11(dropout=None, generate_gt=None, noise=None):
    gt_path = Path.joinpath(Path(__file__).parent, "s03_c010-gt.txt")

    gt_annons = read_annotations(str(gt_path))
    predict_annons = read_annotations(str(gt_path))

    # Without any change
    mapp, miou, _ = mAP(predict_annons, gt_annons)
    print(f"Without any change: mAP {mapp} - mIOU {miou}")

    mapps = []
    mious = []
    # Random Dropout
    for dropout in np.arange(0.0, 0.9, 0.1):
        rgt_annons = copy.deepcopy(gt_annons)
        np.random.shuffle(rgt_annons)
        rgt_annons = rgt_annons[int(len(rgt_annons)*dropout):-1]

        mapp, miou, _ = mAP(gt_annons, rgt_annons)
        mapps.append(mapp)
        mious.append(miou)

        print(f"Dropout of {dropout}: mAP {mapp} - mIOU {miou}")

    dropouts = np.arange(0.0, 0.9, 0.1)
    generate_noise_plot(dropouts, mapps, dropouts, mious, 'mAP', 'mIoU', '% of Dropout', 'Result of mAP/mIoU', 'Applying different percentages of dropout.', f'{OUTPUT_FOLDER}/dropout.png')

    mapps = []
    mious = []
    # Generate Gt
    for generate in np.arange(0.0, 0.9, 0.1):
        rgt_annons = copy.deepcopy(gt_annons)
        np.random.shuffle(rgt_annons)

        extra = rgt_annons[0:int(len(rgt_annons)*generate)]
        rgt_annons += extra
        np.random.shuffle(rgt_annons)

        mapp, miou, _ = mAP(gt_annons, rgt_annons)
        mapps.append(mapp)
        mious.append(miou)

        print(f"With a generation of {generate}%: mAP {mapp} - mIOU {miou}")

    generates = np.arange(0.0, 0.9, 0.1)
    generate_noise_plot(generates, mapps, generates, mious, 'mAP', 'mIoU', '% Generate of new GT boxes', 'Result of mAP/mIoU', 'Applying different percentages of generation new Gt bbox.', f'{OUTPUT_FOLDER}/generation.png')

    # Apply std
    for mean in np.arange(0.0, 161.0, 5):

        mapps = []
        mious = []
        for std in np.arange(10.0, 100.0, 10.0):
            rgt_annons = copy.deepcopy(gt_annons)
            np.random.shuffle(rgt_annons)

            for rgt_annon in rgt_annons:
                random1 = np.random.normal(mean, 2*std) - mean
                rgt_annon.width += (random1) 
                random2 = np.random.normal(mean, 2*std) - mean
                rgt_annon.height += (random2)




            mapp, miou, _ = mAP(gt_annons, rgt_annons)
            mapps.append(mapp)
            mious.append(miou)

            print(f"Noise of std dev {std} and mean: {mean}: mAP {mapp} - mIOU {miou}")

        stds = np.arange(10.0, 100.0, 10.0)
        generate_noise_plot(stds, mapps, stds, mious, 'mAP', 'mIoU', 'Std Dev.', 'Result of mAP/mIoU', f'Applying different Std. Dev with MEAN = {mean} on size', f'{OUTPUT_FOLDER}/size-noise/noise-mean-{mean}-std-dev-size.png')
        
    # Apply std
    for mean in np.arange(0.0, 161.0, 5):

        mapps = []
        mious = []
        for std in np.arange(10.0, 100.0, 10.0):
            rgt_annons = copy.deepcopy(gt_annons)
            np.random.shuffle(rgt_annons)

            for rgt_annon in rgt_annons:
                random1=np.random.normal(mean, 2*std) - mean
                rgt_annon.left += (random1) 
                random2= np.random.normal(mean, 2*std) - mean
                rgt_annon.top += (random2) 

            mapp, miou, _ = mAP(gt_annons, rgt_annons)
            mapps.append(mapp)
            mious.append(miou)

            print(f"Noise of std dev {std} and mean: {mean}: mAP {mapp} - mIOU {miou}")

        stds = np.arange(10.0, 100.0, 10.0)
        generate_noise_plot(stds, mapps, stds, mious, 'mAP', 'mIoU', 'Std Dev.', 'Result of mAP/mIoU', f'Applying different Std. Dev with MEAN = {mean} on location', f'{OUTPUT_FOLDER}/location-noise/noise-mean-{mean}-std-dev-location.png')

def task11_video():
    gt_path = Path.joinpath(Path(__file__).parent, "s03_c010-gt.txt")

    gt_annons = read_annotations(str(gt_path))
    predict_annons = read_annotations(str(gt_path))

    # Apply std
    rgt_annons = copy.deepcopy(gt_annons)
    np.random.shuffle(rgt_annons)

    for rgt_annon in rgt_annons:
        rgt_annon.left += np.random.normal(0, 60)
        rgt_annon.top += np.random.normal(0, 60)
        #rgt_annon.width += np.random.normal(mean, std)
        #rgt_annon.height += np.random.normal(mean, std)

    mapp, miou, frames_miou = mAP(predict_annons, rgt_annons)

    print(f"Noise of std dev {60} and mean: {0}: mAP {mapp} - mIOU {miou}")

    frames = list(range(450, len(frames_miou)))[:150]
    mious = list(frames_miou.values())[450:600]

    generate_video(str(Path.joinpath(Path(__file__).parent, "vdo.avi")), frames, mious, predict_annons, rgt_annons, f'Applying noise to GT Bbox (mean = 0, std dev = 60)', f'noise-mean-{0}-stddev-{60}.gif')


def task12():
    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-mask_rcnn.txt")))
    mapp, _, _ = mAP(gt_annons, predict_annons)
    print(f"Mask rcnn mAP: {mapp}")

    predict_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-annotation.xml")))
    rcnn_mapp = 0
    ssd2_mapp = 0
    yolo3_mapp = 0

    print(f"Getting mAP average for {iters} runs")
    for i in range(iters):

        print(f"|_______ Iteration {i+1}")

        gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-mask_rcnn.txt")))
        tmp_rcnn_mapp, _ = mAP(gt_annons, predict_annons)
        rcnn_mapp = rcnn_mapp + tmp_rcnn_mapp
        print(f"\t|__ Mask rcnn mAP: {tmp_rcnn_mapp:.4f}")

        gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-ssd512.txt")))
        tmp_ssd2_mapp, _ = mAP(gt_annons, predict_annons)
        ssd2_mapp = ssd2_mapp + tmp_ssd2_mapp
        print(f"\t|__ SSD 512 mAP: {tmp_ssd2_mapp:.4f}")

        gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-yolo3.txt")))
        tmp_yolo3_mapp, _ = mAP(gt_annons, predict_annons)
        yolo3_mapp = yolo3_mapp + tmp_yolo3_mapp
        print(f"\t|__ Yolo 3 mAP: {tmp_yolo3_mapp:.4f}")

        avg_rcnn = rcnn_mapp / i+1
        avg_ssd = ssd2_mapp / i+1
        avg_yolo3 = yolo3_mapp / i+1

        print(f"\t|__ Current avg mAPs: RCNN: {avg_rcnn:.4f} | SSD 512: {avg_ssd:.4f} | Yolo3: {avg_yolo3:.4f}")

    avg_rcnn = rcnn_mapp / iters
    avg_ssd = ssd2_mapp / iters
    avg_yolo3 = yolo3_mapp / iters
    print("###########################################################################################")
    print(f"Average mAPs in {iters} iterations: RCNN: {avg_rcnn:.4f} | SSD 512: {avg_ssd:.4f} | Yolo3: {avg_yolo3:.4f}")


def task12_generate_plots():
    predict_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-annotation.xml")))

    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-mask_rcnn.txt")))
    mapp, _, frames_miou = mAP(gt_annons, predict_annons)

    frames = list(range(0, len(frames_miou)))
    mious = list(frames_miou.values())
    generate_frame_mious_plot(frames, mious, "rcnn-plot.png")

    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-ssd512.txt")))
    mapp, _, frames_miou = mAP(gt_annons, predict_annons)

    frames = list(range(0, len(frames_miou)))
    mious = list(frames_miou.values())
    generate_frame_mious_plot(frames, mious, "ssd-plot.png")

    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-yolo3.txt")))
    mapp, _, frames_miou = mAP(gt_annons, predict_annons)

    frames = list(range(0, len(frames_miou)))
    mious = list(frames_miou.values())
    generate_frame_mious_plot(frames, mious, "yolo3-plot.png")

def task12_generate_video():
    predict_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-annotation.xml")))

    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-mask_rcnn.txt")))
    mapp, _, frames_miou = mAP(gt_annons, predict_annons)
    print(f"Mask rcnn mAP: {mapp}")

    frames = list(range(450, len(frames_miou)))[:150]
    mious = list(frames_miou.values())[450:600]

    generate_video(str(Path.joinpath(Path(__file__).parent, "vdo.avi")), frames, mious, gt_annons, predict_annons, 'Mask R-CNN', 'rcnn.gif')

    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-ssd512.txt")))
    mapp, _, frames_miou = mAP(gt_annons, predict_annons)
    print(f"SSD 512 mAP: {mapp}")

    frames = list(range(450, len(frames_miou)))[:150]
    mious = list(frames_miou.values())[450:600]

    generate_video(str(Path.joinpath(Path(__file__).parent, "vdo.avi")), frames, mious, gt_annons, predict_annons, 'SSD 512', 'ssd.gif')

    gt_annons = read_annotations(str(Path.joinpath(Path(__file__).parent, "s03_c010-yolo3.txt")))
    mapp, _, frames_miou = mAP(gt_annons, predict_annons)
    print(f"Yolo 3 mAP: {mapp}")

    frames = list(range(450, len(frames_miou)))[:150]
    mious = list(frames_miou.values())[450:600]

    generate_video(str(Path.joinpath(Path(__file__).parent, "vdo.avi")), frames, mious, gt_annons, predict_annons, 'Yolo 3', 'yolo.gif')


def task13_4():
    pred_000045_10 = read_flow_img(str(Path.joinpath(Path(__file__).parent, "pred_000045_10.png")))
    pred_000157_10 = read_flow_img(str(Path.joinpath(Path(__file__).parent, "pred_000157_10.png")))
    gt_000045_10 = read_flow_img(str(Path.joinpath(Path(__file__).parent, "gt_000045_10.png")))
    gt_000157_10 = read_flow_img(str(Path.joinpath(Path(__file__).parent, "gt_000157_10.png")))

    # -- Frame 45 -- #
    mse_45, error_45, msen_45, pepn_45 = calc_optical_flow(gt_000045_10, pred_000045_10)
    print(msen_45, pepn_45)

    # -- Error -- #
    plt.figure()
    plt.imshow(mse_45)
    plt.title('Error_Flow-45')
    plt.axis('off')
    plt.savefig(os.path.join('results/week1/', 'error_45.png'))
    plt.close()

    # -- Histogram -- #
    plt.figure()
    plt.title('Error Histogram-45')
    plt.hist(error_45, 25, color="orange")
    plt.axvline(msen_45, color='g', linestyle='dashed', linewidth=1, label=f'MSEN {round(msen_45, 1)}')
    plt.legend()
    plt.savefig(os.path.join('results/week1/', 'histogram_45.png'))
    plt.close()

    # -- Mgnitude Flow -- #
    magnitude_gt_45 = magnitude_flow(gt_000045_10)
    plt.imshow(magnitude_gt_45, cmap='hot')
    plt.axis('off')
    plt.title('Magnitude GT Flow 45')
    plt.savefig(os.path.join('results/week1', 'Magnitude GT Flow 45.png'))
    plt.close()

    magnitude_45_pred = magnitude_flow(pred_000045_10)
    plt.imshow(magnitude_45_pred, cmap='hot')
    plt.axis('off')
    plt.title('Magnitude Pred Flow 45')
    plt.savefig(os.path.join('results/week1', 'Magnitude Pred Flow 45.png'))
    plt.close()

    # -- Arrow T4-- #
    img_45_gray = cv2.imread('000045_10_gray.png', cv2.IMREAD_GRAYSCALE)
    plot_arrows(img_45_gray, gt_000045_10, 10, 45, 'GT')
    plot_arrows(img_45_gray, pred_000045_10, 10, 45, 'PRED')

    # -- Frame 157 -- #
    mse_157, error_157, msen_157, pepn_157 = calc_optical_flow(gt_000157_10, pred_000157_10)
    print(msen_157, pepn_157)

    # -- Error -- #
    plt.figure()
    plt.imshow(mse_157)
    plt.title('Error_Flow-157')
    plt.axis('off')
    plt.savefig(os.path.join('results/week1/', 'error_157.png'))
    plt.close()

    # -- Histogram -- #
    plt.figure()
    plt.title('Error Histogram-157')
    plt.hist(error_157, 25, color="orange")
    plt.axvline(msen_157, color='g', linestyle='dashed', linewidth=1, label=f'MSEN {round(msen_157, 1)}')
    plt.legend()
    plt.savefig(os.path.join('results/week1', 'histogram_157.png'))
    plt.close()

    # -- Mgnitude Flow -- #
    magnitude_gt_157 = magnitude_flow(gt_000157_10)
    plt.imshow(magnitude_gt_157, cmap='hot')
    plt.axis('off')
    plt.title('Magnitude GT Flow 157')
    plt.savefig(os.path.join('results/week1', 'Magnitude GT Flow 157.png'))
    plt.close()

    magnitude_157_pred = magnitude_flow(pred_000157_10)
    plt.imshow(magnitude_157_pred, cmap='hot')
    plt.axis('off')
    plt.title('Magnitude Pred Flow 157')
    plt.savefig(os.path.join('results/week1', 'Magnitude Pred Flow 157.png'))
    plt.close()

    # -- Arrow T4-- #
    img_157_gray = cv2.imread('000157_10_gray.png', cv2.IMREAD_GRAYSCALE)
    plot_arrows(img_157_gray, gt_000157_10, 10, 157, 'GT')
    plot_arrows(img_157_gray, pred_000157_10, 10, 157, 'PRED')


task11()
