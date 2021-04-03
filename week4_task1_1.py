import cv2
import numpy as np

from pathlib import Path
from src.block_matching import block_matching
from src.metrics.optical_flow import evaluate_flow
from src.test_block import block_matching_flow

def task1_1():
    previous_frame = cv2.imread(str(Path.joinpath(Path(__file__).parent, './data/000045_10.png')), cv2.IMREAD_GRAYSCALE)
    current_frame = cv2.imread(str(Path.joinpath(Path(__file__).parent, './data/000045_11.png')), cv2.IMREAD_GRAYSCALE)
    flow_noc = cv2.imread(str(Path.joinpath(Path(__file__).parent, './data/gt_000045_10.png')))

    flow = block_matching(current_frame, previous_frame, True, 16, 32)

    print(evaluate_flow(flow_noc, flow))

    flow = block_matching_flow(previous_frame, current_frame, 16, 32)

    print(evaluate_flow(flow_noc, flow))

    #cv2.imwrite('test.png', flow)

if __name__ == "__main__":
    task1_1()