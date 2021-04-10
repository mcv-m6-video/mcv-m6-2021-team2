from pathlib import Path
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.metrics.map import mAP


from src.video import get_frames_from_video, generate_video


def train():
    video_path = str(Path.joinpath(Path(__file__).parent, 'aic19-track1-mtmc-train/train/S03/c010/vdo.avi'))

    """
    for frame_idx, frame in get_frames_from_video(video_path=video_path):
        pass
    """

    detection_path = str(Path.joinpath(Path(__file__).parent, 'aic19-track1-mtmc-train/train/S03/c010/det/det_mask_rcnn.txt'))
    gt_path = str(Path.joinpath(Path(__file__).parent, 'aic19-track1-mtmc-train/train/S03/c010/gt/gt.txt'))


    det_reader = AICityChallengeAnnotationReader(detection_path)
    det_annotations = det_reader.get_annotations(classes=['car'])

    gt_reader = AICityChallengeAnnotationReader(gt_path)
    gt_annotations = gt_reader.get_annotations(classes=['car'])

    generate_video(video_path, 'example.gif', det_annotations, gt_annotations, 'dummy', 218, 268)

if __name__ == "__main__":
    train()