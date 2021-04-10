
def generate_model(video_path: str,
                   perctatge_use_frames: float,
                   grayscale: bool = True) -> Tuple[np.array, np.array]:

    video_frames, frame_width, frame_height = get_frames_from_video(video_path)

    if perctatge_use_frames < 0.0 or perctatge_use_frames > 1.0:
        raise ValueError("The percentatge use of frames should be [0,1].")

    num_frames_to_use = int(len(video_frames)*perctatge_use_frames)

    mean_model_background = np.zeros((frame_height, frame_width))
    variance_model_background = np.zeros((frame_height, frame_width))

    if num_frames_to_use > 0:
        mean_model_background = np.mean(video_frames[:num_frames_to_use], axis=0)
        variance_model_background = np.std(video_frames[:num_frames_to_use], axis=0)

    return mean_model_background, variance_model_background

def get_multi_model(video_path: str,
                   perctatge_use_frames: float,
                   colorspace: str = 'rgb') -> Tuple[np.array, np.array]:
    video_frames, frame_width, frame_height = get_frames_from_video(video_path, False, colorspace=colorspace)

    if perctatge_use_frames < 0.0 or perctatge_use_frames > 1.0:
        raise ValueError("The percentatge use of frames should be [0,1].")

    num_frames_to_use = int(len(video_frames)*perctatge_use_frames)

    mean_model_background = np.zeros((frame_height, frame_width, 3))
    variance_model_background = np.zeros((frame_height, frame_width, 3))

    if num_frames_to_use > 0:
        mean_model_background = np.mean(video_frames[:num_frames_to_use], axis=0)
        variance_model_background = np.std(video_frames[:num_frames_to_use], axis=0)

    return mean_model_background, variance_model_background
