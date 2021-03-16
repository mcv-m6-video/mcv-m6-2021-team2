from typing import List
from pathlib import Path
import xml.etree.ElementTree as ET

from src.annotation import Annotation

def read_xml(path: str, include_parked: bool = False) -> List[Annotation]:
    if not Path(path).exists():
        raise FileNotFoundError(f"The file pat provided: {path} does not exists.")

    annons = []

    for elem in ET.parse(path).getroot():
        if elem.tag == 'track' and elem.attrib['label'] == 'car':
            for box in elem:
                if (box[0].text == "true" and include_parked) or box[0].text == "false":
                    box_attrib = box.attrib
                    annons.append(Annotation(
                        frame=int(box_attrib['frame'])+1,
                        left=float(box_attrib['xtl']),
                        top=float(box_attrib['ytl']),
                        width=float(box_attrib['xbr']),
                        height=float(box_attrib['ybr']),
                        label='car'
                    ))

    return annons

def read_txt(path: str) -> List[Annotation]:
    if not Path(path).exists():
        raise FileNotFoundError(f"The file pat provided: {path} does not exists.")

    annons = []

    with open(path, 'r') as f:
        for line in f:
            params = line.split(',')

            annons.append(Annotation(
                frame=int(params[0]),
                left=float(params[2]),
                top=float(params[3]),
                width=float(params[2])+float(params[4]),
                height=float(params[3])+float(params[5]),
                score=float(params[6]),
                label='car'
            ))

    return annons