import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from src.annotation import Annotation

def read_annotations(file_path : str) -> List[Annotation]:

    if not Path(file_path).exists():
        raise FileNotFoundError(f"The file pat provided: {file_path} does not exists.")

    annons = []

    if file_path.endswith(".xml"):
        """
        Expected xml file structure:
        <annotations>
            <version></version>
            <meta></meta>
            <track id="x" label="car/bike">
                <box frame="" xtl="" ytl="" xbr="" ybr="" outside="" occluded="" keyframe="">
                    <attribute name="parked">false</attribute> <-- only appears with cars
                </box>
                ...
            </track>
            ...
        </annotations>
        """
        for elem in ET.parse(file_path).getroot():
            if elem.tag == 'track' and elem.attrib['label'] == 'car':
                for box in elem:
                    box_attrib = box.attrib
                    annons.append(Annotation(
                        frame=int(box_attrib['frame']),
                        left=float(box_attrib['xtl']),
                        top=float(box_attrib['ytl']),
                        width=float(box_attrib['xbr']),
                        height=float(box_attrib['ybr']),
                        name='car'
                    ))
        return annons
    elif file_path.endswith('.txt'):
        """
        Expected txt content: [frame, ID, left, top, width, height, 1, -1, -1, -1]
        """
        with open(file_path, 'r') as f:
            for line in f:
                params = line.split(',')

                annons.append(Annotation(
                    frame=int(params[0]),
                    left=float(params[2]),
                    top=float(params[3]),
                    width=float(params[2])+float(params[4]),
                    height=float(params[3])+float(params[5]),
                    name='car'
                ))
        return annons
    else:
        raise NotImplementedError(f"The extension file is not supported.")