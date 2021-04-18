from src.metric_learning.build_dataset import generate_metric_learning_database
from src.metric_learning.train import train
from src.metric_learning.embeddings import extract_embeddings, plot_embeddings
from pathlib import Path

"""
generate_metric_learning_database(dataset_path=str(Path.joinpath(Path(__file__).parent, './data/AICity_track_data/train').absolute()),
                                  train_seq=['S01', 'S04'],
                                  test_seq=['S03'],
                                  img_size=128)
"""


train(output_path=str(Path.joinpath(Path(__file__).parent, './checkpoints')),
      epochs=10,
      lr=0.0001,
      batch_size=255,
      num_workers=12)


"""
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.metric_learning.train import get_transform
from matplotlib import pyplot as plt
import torch

dataset = ImageFolder(root=f'./data/ml_database/train', transform=get_transform(train=False))
dataloader = DataLoader(dataset, batch_size=255, shuffle=False, num_workers=12)

model = torch.load('./checkpoints/epoch_0_ckpt.pth')

embeds, labels = extract_embeddings(model, dataloader)

print(embeds[0])
print(labels)
print(len(embeds))
print(len(labels))
"""

#plot_embeddings(model, dataloader, max_classes=10)
#plt.savefig(f"embedding_train.png")
