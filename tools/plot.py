import matplotlib
import matplotlib.pyplot as plt
import statistics 


def generate_frame_mious_plot(frames, mious, output_path):
    
    fig, ax = plt.subplots()
    ax.plot(frames, mious)

    ax.set(xlabel='frames', ylabel='mIoU', title='mIoU x Frame')
    fig.savefig(output_path)

    zeroDatapoint = 1
    minDatapoint = 1
    targetframe = 0
    targetZeroFrame = 0

    for datapoint, frame in zip(mious, frames):
        if datapoint < minDatapoint:
            if datapoint != 0:
                zeroDatapoint = datapoint
                targetZeroFrame = frame
            else:
                minDatapoint = datapoint
                targetframe = frame
    print(f'Generated plot into {output_path}')
    print(f"\nMin miou: {minDatapoint} at frame: {targetframe}")
    print(f"Zero miou: {zeroDatapoint} at frame: {targetZeroFrame}")
    print(f"Standard deviation: {statistics.stdev(mious)}")
    print(f"Mean: {statistics.mean(mious)}")
    print("_______________________________________________________")
    