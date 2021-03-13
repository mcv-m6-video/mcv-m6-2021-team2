from pathlib import Path

from src.readers.ai_city_reader import read_txt, read_xml
from src.metrics.map import mAP

gt = read_xml(Path.joinpath(Path(__file__).parent, './data/week1/s03_c010-annotation.xml'), True)
mask_rcnn = read_txt(Path.joinpath(Path(__file__).parent, './data/week1/s03_c010-mask_rcnn.txt'), True)

print(mAP(mask_rcnn, gt, ['car'], True))