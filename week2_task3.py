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

mog_substractor  = cv2.bgsegm.createBackgroundSubtractorMOG(nmixtures=2, history=200)
mog2_substractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=200)
lsbp_substractor = cv2.bgsegm.createBackgroundSubtractorLSBP(minCount=5, mc=cv2.bgsegm.LSBP_CAMERA_MOTION_COMPENSATION_NONE)

mog_bboxes_all_frames  = OrderedDict()
mog2_bboxes_all_frames = OrderedDict()
lsbp_bboxes_all_frames = OrderedDict()

FIRST_FRAME_ID = 535
total_files =len(frame_files)
print(f"Found {total_files} frames.")


gt_bounding_boxes_orig = read_xml(str(FULL_ANNOTATION_PATH), include_parked=False)

gt_bounding_boxes = list(filter(lambda x: x.frame >= FIRST_FRAME_ID, gt_bounding_boxes_orig))

# gt_bounding_boxes = gt_bounding_boxes_orig[FIRST_FRAME_ID:]

# print(len(gt_bounding_boxes_orig))
print(len(gt_bounding_boxes))


if (Path('preanotation.pickle').exists()):
    print("Found preanotation pkl, loading")
    with open('preanotation.pickle', 'rb') as f:
        mog_bboxes_all_frames, mog2_bboxes_all_frames, lsbp_bboxes_all_frames=  pickle.load(f)

else:
    for i, frame_file in enumerate(frame_files[FIRST_FRAME_ID:], FIRST_FRAME_ID):




        frame = asarray(frame_file)

        frame_bbox = frame.copy()
        frame_bbox = cv2.cvtColor(frame_bbox,cv2.COLOR_GRAY2RGB)
        frame_mog = frame_bbox.copy()
        frame_mog2 = frame_bbox.copy()
        frame_lsbp = frame_bbox.copy()

        for item in gt_bounding_boxes:
            if item.frame == i:
                gtx1, gty1, gtx2, gty2 = item.get_bbox()
                gtx1, gty1, gtx2, gty2 = int(gtx1), int(gty1), int(gtx2), int(gty2)
                # print(f"{gtx1}, {gty1}, {gtx2}, {gty2}")
                cv2.rectangle(frame_bbox, (gtx1, gty1), (gtx2, gty2), (255,255,0), 2)
                

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

        
        # self.xtl = xtl
        # self.ytl = ytl
        # self.xbr = xbr
        # self.ybr = ybr

        font = cv2.FONT_HERSHEY_PLAIN
        fontsize = 2

        for item in mog_bboxes:
            x1 = item.xtl
            y1 = item.ytl
            x2 = item.xbr
            y2 = item.ybr
            color = (255, 0, 255)
            fontX = x2 - abs(x2-x1)
            fontY = abs(y2-1)
            thickness = 2

            cv2.rectangle(frame_bbox, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bbox, ' MOG Box', (fontX, fontY),
                        font, fontsize, color, thickness, cv2.LINE_AA)

            cv2.rectangle(frame_mog, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_mog, ' MOG Box', (fontX, fontY),
                        font, fontsize, color, thickness, cv2.LINE_AA)

        for item in mog2_bboxes:
            x1 = item.xtl
            y1 = item.ytl
            x2 = item.xbr
            y2 = item.ybr
            color = (255, 125, 125)
            fontX = x2 - abs(x2-x1)
            fontY = abs(y2-1)

            cv2.rectangle(frame_bbox, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bbox, ' MOG2 Box', (fontX, fontY),
                        font, fontsize, color, thickness, cv2.LINE_AA)

            cv2.rectangle(frame_mog2, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_mog2, ' MOG2 Box', (fontX, fontY),
                        font, fontsize, color, thickness, cv2.LINE_AA)

        for item in lsbp_bboxes:
            x1 = item.xtl
            y1 = item.ytl
            x2 = item.xbr
            y2 = item.ybr
            color = (125, 125, 255)
            fontX = x2 - abs(x2-x1)
            fontY = abs(y2-1)

            cv2.rectangle(frame_bbox, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bbox, ' LSBP Box', (fontX, fontY),
                        font, fontsize, color, thickness, cv2.LINE_AA)

            cv2.rectangle(frame_lsbp, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_lsbp, ' LSBP Box', (fontX, fontY),
                        font, fontsize, color, thickness, cv2.LINE_AA)


        cv2.imshow("window", frame_bbox)
        cv2.waitKey(1)
 
        cv2.imwrite(f"results/week2/all/frame_{i:07d}.jpg", frame_bbox)
        cv2.imwrite(f"results/week2/mog/frame_{i:07d}.jpg", frame_mog)
        cv2.imwrite(f"results/week2/mog2/frame_{i:07d}.jpg", frame_mog2)
        cv2.imwrite(f"results/week2/lsbp/frame_{i:07d}.jpg", frame_lsbp)
        # cv2.destroyAllWindows()  

        txt=f' working on frame {i} of {total_files}. found {len(mog_bboxes)} bboxes'

        print_percent_done(i-FIRST_FRAME_ID, len(frame_files)-FIRST_FRAME_ID, title=txt)

    with open('preanotation.pickle', 'wb') as f:
            pickle.dump((mog_bboxes_all_frames, mog2_bboxes_all_frames, lsbp_bboxes_all_frames), f)

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
