import cv2
from src.utils.entities import BoundingBox

def remove_noise_from_mask(mask):
    # To close little holes in the masks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # To remove noise around the masks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return mask

def detect_bbox_from_mask(mask, frame_id, min_width=120, max_width=800, min_height=100, max_height=600):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    num_detections = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if min_width < w < max_width and min_height < h < max_height:
            detections.append(BoundingBox(frame_id, num_detections, 'car', x, y, x + w, y + h))
            num_detections += 1
    return detections