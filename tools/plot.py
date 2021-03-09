import matplotlib
import matplotlib.pyplot as plt

def generate_frame_mious_plot(frames, mious, output_path):
    fig, ax = plt.subplots()
    ax.plot(frames, mious)

    ax.set(xlabel='frames', ylabel='mIoU', title='mIoU x Frame')
    fig.savefig(output_path)
    print(f'Generating plot into {output_path}')