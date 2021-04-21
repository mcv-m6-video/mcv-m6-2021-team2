"""
Embeddings from: https://github.com/pytorch/vision/tree/master/references/similarity
"""

import io

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision.transforms.functional import to_tensor


@torch.no_grad()
def extract_embeddings(model, loader):
    model.eval()
    embeds, labels = [], []

    for images, _labels in loader:
        if torch.cuda.is_available():
            images = images.cuda()
        out = model.get_embedding(images).cpu().numpy()
        embeds.append(out)
        labels.append(_labels.numpy())

    embeds = np.vstack(embeds)
    labels = np.concatenate(labels)

    return embeds, labels


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    image = to_tensor(image)
    return image


def plot_embeddings(model, loader, output_path, max_classes=None):
    embeds, labels = extract_embeddings(model, loader)

    embeds = TSNE(n_components=2, verbose=1).fit_transform(embeds)

    figure = plt.figure(figsize=(10, 10))
    if max_classes is not None:
        selected_classes = np.random.choice(loader.dataset.classes, max_classes)
    else:
        selected_classes = loader.dataset.classes
    for cls in selected_classes:
        idx = loader.dataset.class_to_idx[cls]
        inds = labels == idx
        #cmap = plt.cm.get_cmap("tab20c")
        #colors = cmap(np.arange(len(embeds[inds, 1])))
        plt.scatter(embeds[inds, 0], embeds[inds, 1], alpha=0.5)

    plt.legend(loader.dataset.classes)
    plt.savefig(output_path)
    plt.close()