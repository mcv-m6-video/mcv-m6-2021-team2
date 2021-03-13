from src.annotation import Annotation

def IoU(annotation_a: Annotation, annotation_b: Annotation) -> float:
    bbox_a = annotation_a.get_bbox()
    bbox_b = annotation_b.get_bbox()

    xA = max(bbox_a[0], bbox_b[0])
    yA = max(bbox_a[1], bbox_b[1])
    xB = min(bbox_a[2], bbox_b[2])
    yB = min(bbox_a[3], bbox_b[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
    boxAArea = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    boxBArea = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou