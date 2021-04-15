from typing import List, OrderedDict
from collections import defaultdict, OrderedDict
from pathlib import Path
from copy import deepcopy

import numpy as np
import xmltodict

from src.detection import Detection


def parse_annotations_from_xml(path: str) -> List[Detection]:
    with open(path) as f:
        tracks = xmltodict.parse(f.read())['annotations']['track']

    annotations = []

    for track in tracks:
        id = track['@id']
        label = track['@label']
        boxes = track['box']

        for box in boxes:
            if label == 'car':
                parked = box['attribute']['#text'].lower() == 'true'
            else:
                parked = None

            annotations.append(Detection(
                frame=int(box['@frame']),
                id=int(id),
                label=label,
                xtl=float(box['@xtl']),
                ytl=float(box['@ytl']),
                xbr=float(box['@xbr']),
                ybr=float(box['@ybr']),
                parked=parked
            ))

    return annotations


def parse_annotations_from_txt(path: str):
    with open(path) as f:
        lines = f.readlines()

    annotations = []

    for line in lines:
        data = line.split(',')

        annotations.append(Detection(
            frame=int(data[0])-1,
            id=int(data[1]),
            label='car',
            xtl=float(data[2]),
            ytl=float(data[3]),
            xbr=float(data[2])+float(data[4]),
            ybr=float(data[3])+float(data[5]),
            score=float(data[6])
        ))

    return annotations

def group_by_frame(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det.frame].append(det)
    return OrderedDict(sorted(grouped.items()))


def group_by_id(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det.id].append(det)
    return OrderedDict(sorted(grouped.items()))


def group_in_tracks(detections, camera):
    grouped = group_by_id(detections)
    tracks = {}
    for id in grouped.keys():
        tracks[id] = Track(id, grouped[id], camera)
    return tracks