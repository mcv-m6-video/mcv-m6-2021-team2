from pathlib import Path
from src.readers.ai_city_reader import AICityChallengeAnnotationReader
from src.metrics.map import mAP


from src.video import get_frames_from_video


def train():
    video_path = str(Path.joinpath(Path(__file__).parent, 'aic19-track1-mtmc-train/train/S03/c010/vdo.avi'))

    for frame_idx, frame in get_frames_from_video(video_path=video_path, start_frame=4, end_frame=10):
        print(frame.shape)

if __name__ == "__main__":
    train()