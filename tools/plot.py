import matplotlib
import matplotlib.pyplot as plt

def generate_frame_mious_plot(frames, mious, output_path):
    fig, ax = plt.subplots()
    ax.plot(frames, mious)

    ax.set(xlabel='frames', ylabel='mIoU', title='mIoU x Frame')
    fig.savefig(output_path)
    print(f'Generating plot into {output_path}')

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