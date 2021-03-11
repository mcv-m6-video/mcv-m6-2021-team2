from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import cv2
import xmltodict

from src.utils.entities import BoundingBox

class AnnotationReader():

    def __init__(self, annotation_path):
        self.annotations = self.parse_annotations(annotation_path)
        self.classes = np.unique([bb.label for bb in self.annotations])

    def parse_annotations(self, annotation_path):
        extension = annotation_path.suffix
        if extension == '.xml':
            annotations = self._parse_xml(annotation_path)
        elif extension == '.txt':
            annotations = self._parse_txt(annotation_path)
        else:
            raise ValueError(f'The file passed has not the correct extension: {extension}')
        return annotations
    
    def _parse_xml(self, annotation_path):
        with open(annotation_path) as file:
            tracks = xmltodict.parse(file.read())['annotations']['track']

        annotations = []
        for track in tracks:
            instance_id = track['@id']
            label = track['@label']
            boxes = track['box']
            for box in boxes:
                annotations.append(BoundingBox(
                    frame=int(box['@frame']),
                    instance_id=int(instance_id),
                    label=label,
                    xtl=float(box['@xtl']),
                    ytl=float(box['@ytl']),
                    xbr=float(box['@xbr']),
                    ybr=float(box['@ybr']),
                    score=None
                ))

        return annotations

    def _parse_txt(self, annotation_path):
        """ MOTChallenge format [frame, ID, left, top, width, height, conf, -1, -1, -1] """
        with open(annotation_path) as file:
            lines = file.readlines()

        annotations = []
        for line in lines:
            data = line.split(',')
            annotations.append(BoundingBox(
                frame=int(data[0])-1,
                instance_id=int(data[1]),
                label='car',
                xtl=float(data[2]),
                ytl=float(data[3]),
                xbr=float(data[2])+float(data[4]),
                ybr=float(data[3])+float(data[5]),
                score=float(data[6])
            ))

        return annotations

    def get_bounding_boxes(self, classes=None, group_by='frame'):
        """ Returns: bounding_boxes: {frame: [BoundingBox,...]} if group_by='frame' """
        if classes is None:
            classes = self.classes
        
        bounding_boxes = []
        for bb in self.annotations:
            if bb.label in classes:  # filter by specified class
                bounding_boxes.append(bb)

        if group_by:
            bounding_boxes = self._group_by(bounding_boxes, group_by)
        
        return bounding_boxes

    def _group_by(self, bounding_boxes, group_by):
        grouped = defaultdict(list)
        for bb in bounding_boxes:
            if group_by == 'frame':
                grouped[bb.frame].append(bb)
            elif group_by == 'id':
                grouped[bb.id].append(bb)
            else:
                raise ValueError(f'This group by filter is not available: {group_by}')
        return OrderedDict(sorted(grouped.items()))