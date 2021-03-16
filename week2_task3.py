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

        

def bounding_boxes(mask, frame_id, min_height=100, max_height=600, min_width=120, max_width=800):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_width < w < max_width and min_height < h < max_height:
            detections.append(BoundingBox(frame_id, None, 'car', x, y, x + w, y + h))
    return detections



frame_files, frame_h, frame_w = get_frames_from_video(path=str(VIDEO_PATH), grayscale=True)

mog_substractor  = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2_substractor = cv2.createBackgroundSubtractorMOG2()
lsbp_substractor = cv2.bgsegm.createBackgroundSubtractorLSBP()

mog_bboxes_all_frames  = OrderedDict()
mog2_bboxes_all_frames = OrderedDict()
lsbp_bboxes_all_frames = OrderedDict()

FIRST_FRAME_ID = 535
total_files =len(frame_files)
print(f"Found {total_files} frames.")

if (Path('preanotation.pickle').exists()):
    print("Found preanotation pkl, loading")
    with open('preanotation.pickle', 'rb') as f:
        mog_bboxes_all_frames, mog2_bboxes_all_frames, lsbp_bboxes_all_frames=  pickle.load(f)

else:
    for i, frame_file in enumerate(frame_files[FIRST_FRAME_ID:], FIRST_FRAME_ID):

        frame = asarray(frame_file)
        mog_mask = mog_substractor.apply(frame)
        mog2_mask = mog2_substractor.apply(frame)
        lsbp_mask = lsbp_substractor.apply(frame)
        
        # print(f' unique {np.unique(mog_mask)}')
        # print(f' unique {np.unique(mog2_mask)}')
        # print(f' unique {np.unique(lsbp_mask)}')

        mog_bboxes = bounding_boxes(mog_mask, i)
        mog2_bboxes = bounding_boxes(mog2_mask, i)
        lsbp_bboxes = bounding_boxes(lsbp_mask, i)


        mog_bboxes_all_frames[i] = mog_bboxes
        mog2_bboxes_all_frames[i] = mog2_bboxes
        lsbp_bboxes_all_frames[i] = lsbp_bboxes

        txt=f' working on frame {i} of {total_files}'

        print_percent_done(i-FIRST_FRAME_ID, len(frame_files)-FIRST_FRAME_ID, title=txt)

    with open('preanotation.pickle', 'wb') as f:
            pickle.dump((mog_bboxes_all_frames, mog2_bboxes_all_frames, lsbp_bboxes_all_frames), f)

gt_bounding_boxes = read_xml(str(FULL_ANNOTATION_PATH), include_parked=True)



a = list(itertools.chain.from_iterable(mog_bboxes_all_frames.values()))
b = list(itertools.chain.from_iterable(mog2_bboxes_all_frames.values()))
c = list(itertools.chain.from_iterable(lsbp_bboxes_all_frames.values()))

# typetext = f"gt type: {type(gt_bounding_boxes[0])} b type: {type(gt_bounding_boxes[0])} b type: {type(b[0])} c type: {type(c[0])} "
# print(typetext)
with open('mog.pickle', 'wb') as f:
        pickle.dump(a, f)
with open('mog2.pickle', 'wb') as f:
        pickle.dump(b, f)
with open('lsbp.pickle', 'wb') as f:
        pickle.dump(c, f)

cl = ['car']
mog_map, _, _, mog_mious_per_class = mAP(a, gt_bounding_boxes, classes=cl, score_available=False)
mog2_map, _, _, mog2_mious_per_class = mAP(b, gt_bounding_boxes, classes=cl, score_available=False)
lsbp_map, _, _, lsbp_mious_per_class = mAP(b, gt_bounding_boxes, classes=cl, score_available=False)

print(f' MOG mAP => {mog_map:0.4f}')
print(f' MOG2 mAP => {mog2_map:0.4f}')
print(f' LSBP mAP => {lsbp_map:0.4f}')