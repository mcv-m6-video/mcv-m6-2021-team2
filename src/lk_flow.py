import cv2
import time
import numpy as np


def LK_flow(previous_frame, current_frame):

    height, width = previous_frame.shape[:2]
    # dense flow: one point for each pixel
    p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))

    # params for lucas-kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    tic = time.time()
    p1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, p0, None, **lk_params)
    toc = time.time()

    p0 = p0.reshape((height, width, 2))
    p1 = p1.reshape((height, width, 2))
    st = st.reshape((height, width))

    # flow field computed by subtracting prev points from next points
    flow = p1 - p0
    flow[st == 0] = 0
    return flow, tic, toc
