from typing import List
from collections import defaultdict, OrderedDict
from pathlib import Path
from copy import deepcopy

import numpy as np
import xmltodict

from src.detection import Detection
from src.track import Track


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


def parse_annotations_from_txt(path: str) -> List[Detection]:
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


def parse_annotations(path: str) -> List[Detection]:
    if Path(path).suffix == ".xml":
        return parse_annotations_from_xml(path)
    elif Path(path).suffix == ".txt":
        return parse_annotations_from_txt(path)
    else:
        raise ValueError(f'Invalid file extension: {ext}')


def group_by_frame(detections: List[Detection]):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det.frame].append(det)
    return OrderedDict(sorted(grouped.items()))


def group_by_id(detections: List[Detection]):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det.id].append(det)
    return OrderedDict(sorted(grouped.items()))


def resolve_tracks_from_detections(detections):
    grouped = group_by_id(detections)
    tracks = {}
    for identifier in grouped.keys():
        tracks[identifier] = Track(identifier, grouped[identifier])
    return tracks


class AICityChallengeAnnotationReader:

    def __init__(self, path):
        self.annotations = parse_annotations(path)
        self.classes = np.unique([det.label for det in self.annotations])

    def get_annotations(self, classes=None, noise_params=None, do_group_by_frame=True, only_not_parked=False):
        if classes is None:
            classes = self.classes

        detections = []
        for det in self.annotations:
            if det.label in classes:  # filter by class
                if only_not_parked and det.parked:
                    continue
                d = deepcopy(det)
                if noise_params:  # add noise
                    if np.random.random() > noise_params['drop']:
                        box_noisy = d.bbox + np.random.normal(noise_params['mean'], noise_params['std'], 4)
                        d.xtl = box_noisy[0]
                        d.ytl = box_noisy[1]
                        d.xbr = box_noisy[2]
                        d.ybr = box_noisy[3]
                        detections.append(d)
                else:
                    detections.append(d)

        if do_group_by_frame:
            detections = group_by_frame(detections)

        return detections