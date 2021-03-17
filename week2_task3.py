import cv2
import natsort
import itertools
import os
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw
from numpy import asarray
from collections import OrderedDict
from src.metrics.map import mAP
from pathlib import Path
from src.utils.progress_bar import print_percent_done
from src.video import get_frames_from_video
from src.annotation import Annotation
from src.readers.ai_city_reader import read_xml, read_txt

DATA_ROOT = Path('data')
FULL_ANNOTATION_PATH = DATA_ROOT / 'ai_challenge_s03_c010-full_annotation.xml'
AICITY_DATA_ROOT = DATA_ROOT / Path('AICity_data/train/S03/c010')
FRAMES_LOCATION = DATA_ROOT / 'frames'
RESULTS_ROOT = Path('results')
VIDEO_PATH = AICITY_DATA_ROOT / 'vdo.avi'
ROI_PATH = AICITY_DATA_ROOT / 'roi.jpg'
BG_SUBS_METHOD_LIST = ["CNT","MOG","MOG2","KNN","GSOC","LSBP","GMG"]

assert DATA_ROOT.exists()
assert FULL_ANNOTATION_PATH.exists()
assert AICITY_DATA_ROOT.exists()
assert FRAMES_LOCATION.exists()
assert RESULTS_ROOT.exists()


class BoundingBox():
    def __init__(self, frame, instance_id, label, xtl, ytl, xbr, ybr, score=None, parked=None):
        self.frame = frame
        self.instance_id = instance_id
        self.label = label
        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr
        self.confidence = score
        self.parked = parked

    @property
    def bbox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    @property
    def width(self):
        return abs(self.xbr - self.xtl)

    @property
    def height(self):
        return abs(self.ytl - self.ybr)

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (int((self.xtl + self.xbr) / 2), int((self.ybr + self.ytl) / 2))

    def get_bbox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    def __repr__(self):
        return f'BoundingBox:: frame:{self.frame}, instance_id:{self.instance_id}, label: {self.label}, confidence: {self.confidence}'


def bounding_boxes(mask, frame_id, min_height=10, max_height=600, min_width=76, max_width=800):

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_width < w < max_width and min_height < h < max_height:
            detections.append(BoundingBox(
                frame_id, None, 'car', x, y, x + w, y + h))
    return detections


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    overlay = img.copy()
    output = img.copy()
    alpha = 0.3
    cv2.rectangle(overlay, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(overlay, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

def bg_subs_selector(bg_subs):
    if bg_subs == "MOG":
        substractor = cv2.bgsegm.createBackgroundSubtractorMOG(
            nmixtures=2, history=50)
    elif bg_subs == "MOG2":
        substractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, history=50)
    elif bg_subs == "LSBP":
        substractor = cv2.bgsegm.createBackgroundSubtractorLSBP(
            minCount=5, mc=cv2.bgsegm.LSBP_CAMERA_MOTION_COMPENSATION_NONE)
    elif bg_subs == "GSOC":
        substractor = cv2.bgsegm.createBackgroundSubtractorGSOC(
            nSamples=100, 
            replaceRate=0.009, 
            propagationRate=0.005, 
            noiseRemovalThresholdFacBG=0.004, 
            noiseRemovalThresholdFacFG=0.008)
    elif bg_subs == "KNN":
        substractor = cv2.createBackgroundSubtractorKNN(history=50, 
        dist2Threshold=200, 
        detectShadows=False)
    elif bg_subs == "GMG":
        substractor = cv2.bgsegm.createBackgroundSubtractorGMG(
            initializationFrames=200,
            decisionThreshold=0.7)
    elif bg_subs == "CNT":
        substractor = cv2.bgsegm.createBackgroundSubtractorCNT(
            minPixelStability=2,
            maxPixelStability=30,
            useHistory=False, 
            isParallel=True)
    return substractor

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################



with open('SotA_mAPs.txt', 'a') as file:
    results_line = f'\n\n#################\n#################\n'
    file.write(results_line)

frame_files, frame_h, frame_w = get_frames_from_video(
    path=str(VIDEO_PATH), grayscale=True)


FIRST_FRAME_ID = 535
total_files = len(frame_files)
print(f"Found {total_files} frames.")


gt_bounding_boxes_orig = read_xml(
    str(FULL_ANNOTATION_PATH), include_parked=False)

gt_bounding_boxes = list(
    filter(lambda x: x.frame >= FIRST_FRAME_ID, gt_bounding_boxes_orig))

roi = cv2.imread(str(ROI_PATH), cv2.IMREAD_GRAYSCALE)

for bg_sub_method in BG_SUBS_METHOD_LIST:

    imgpath = f"results/week2/{str(bg_sub_method)}"
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    bg_substractor = bg_subs_selector(bg_sub_method)
    pred_bboxes_all_frames = OrderedDict()
    if (Path(f"{bg_sub_method}_boxes.pickle").exists()):
        print(f"Found {bg_sub_method}_boxes pkl, loading")
        with open(f'{bg_sub_method}_boxes.pickle', 'rb') as f:
            pred_bboxes_all_frames=  pickle.load(f)
    
    else:

        for i, frame_file in enumerate(frame_files[FIRST_FRAME_ID:], FIRST_FRAME_ID):

            ############## PREP ########################
            frame = asarray(frame_file)
            frame_bbox = frame.copy()
            frame_bbox = cv2.cvtColor(frame_bbox, cv2.COLOR_GRAY2RGB)

            ################## DRAWING ###########################
            for item in gt_bounding_boxes:
                if item.frame == i:
                    gtx1, gty1, gtx2, gty2 = item.get_bbox()
                    gtx1, gty1, gtx2, gty2 = int(gtx1), int(
                        gty1), int(gtx2), int(gty2)
                    # print(f"{gtx1}, {gty1}, {gtx2}, {gty2}")
                    cv2.rectangle(frame_bbox, (gtx1, gty1),
                                (gtx2, gty2), (255, 255, 0), 3)

            ################################################################

            
            frame = cv2.GaussianBlur(frame, (7,7), 0)

            mask = bg_substractor.apply(frame)
            mask = mask & roi
            pred_bboxes = bounding_boxes(mask, i)
            pred_bboxes_all_frames[i] = pred_bboxes

            txt_font = cv2.FONT_HERSHEY_PLAIN
            fontsize = 4

            for item in pred_bboxes:
                x1 = item.xtl
                y1 = item.ytl
                x2 = item.xbr
                y2 = item.ybr
                color = (255, 0, 255)
                fontX = x2 - abs(x2-x1) + fontsize*2
                fontY = abs(y2-fontsize*2)
                rect_thickness = 4
                text_thickness = 2

                cv2.rectangle(frame_bbox, (x1, y1), (x2, y2),
                            color, rect_thickness)
                # cv2.putText(frame_bbox, str(bg_sub_method), (fontX, fontY),
                #             font, fontsize, color, text_thickness, cv2.LINE_AA)
                frame_bbox = draw_text(frame_bbox, str(bg_sub_method), font=txt_font, pos=(fontX, fontY), font_scale=fontsize,
                font_thickness=text_thickness, text_color=color, text_color_bg=(0,0,0))


            
            scale_percent = 50 # percent of original size
            width = int(frame_bbox.shape[1] * scale_percent / 100)
            height = int(frame_bbox.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            frame_bbox = cv2.resize(frame_bbox, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(f"{imgpath}/frame_{i:07d}.jpg", frame_bbox)

            txt = f' {bg_sub_method} working on frame {i} of {total_files}. found {len(pred_bboxes)} bboxes'

            print_percent_done(
                i-FIRST_FRAME_ID, len(frame_files)-FIRST_FRAME_ID, title=txt)

            with open(f'{bg_sub_method}_boxes.pickle', 'wb') as f:
                pickle.dump(pred_bboxes_all_frames, f)
    

    cl = ['car']

    pred_bounding_boxes = list(
        itertools.chain.from_iterable(pred_bboxes_all_frames.values()))
    pred_map, _, _, _ = mAP(
        pred_bounding_boxes, gt_bounding_boxes, classes=cl, score_available=False)

    with open('SotA_mAPs.txt', 'a') as file:
        results_line = f'{bg_sub_method} mAP: {pred_map:0.4f} \n'
        file.write(results_line)
