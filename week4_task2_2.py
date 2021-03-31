from vidstab import VidStab

from pathlib import Path


RESULTS_DIR = Path('Results/week4/task2_2')


def stabilize(input_video_path: str, output_video_path: str, kp_detector: str = 'GFTT', **kp_kwargs):
    # Off-the-shelf Stabilization
    stabilizer = VidStab(kp_method=kp_detector, **kp_kwargs)
    stabilizer.stabilize(input_path=input_video_path, output_path=output_video_path)


def task_2_2():
    INPUT_VIDEO_PATH = Path('week4_video.mov')
    params = [
        ('GFTT', []),
        ('BRISK', []),
        ('DENSE', []),
        ('FAST', []),
        ('HARRIS', []),
        ('MSER', []),
        ('ORB', []),
        ('STAR', []),
    ]

    for kp_detector, kp_kwargs in params:
        output_video_path = RESULTS_DIR / f'output_{kp_detector}_{kp_kwargs}.avi'
        stabilize(input_video_path=INPUT_VIDEO_PATH, output_video_path=output_video_path, kp_detector=kp_detector, **kp_kwargs)


if __name__ == '__main__':
    task_2_2()
