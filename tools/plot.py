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

def generate_noise_plot(x, y, xx, yy, label1, label2, xlabel, ylabel, title, output_path):
    plt.plot(x, y, label=label1)
    plt.plot(xx, yy, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f'Generating plot into {output_path}')

    
def generate_noise_plot_gif(x, y, xx, yy, label1, label2, xlabel, ylabel, title, output_path):
    plt.plot(x, y, label=label1)
    plt.plot(xx, yy, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f'Generating plot into {output_path}')