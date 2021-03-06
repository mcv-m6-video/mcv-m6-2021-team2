"""
The following code is taken from: https://github.com/pytorch/vision/blob/master/references/similarity/train.py
All the credits to the original authors.
"""

import os
import argparse
import datetime

from typing import NoReturn
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter

from src.metric_learning.sampler import PKSampler
from src.metric_learning.loss import TripletMarginLoss
from src.metric_learning.network import EmbeddingNet
from src.metric_learning.embeddings import plot_embeddings, plot_to_image


def train_epoch(model, optimizer, criterion, data_loader, epoch, print_freq=20):
    model.train()
    running_loss = 0
    running_frac_pos_triplets = 0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        samples, targets = data[0].cuda(), data[1].cuda()

        embeddings = model(samples)
        loss, frac_pos_triplets = criterion(embeddings, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_frac_pos_triplets += float(frac_pos_triplets)

        if i % print_freq == print_freq - 1:
            i += 1
            avg_loss = running_loss / print_freq
            avg_trip = 100.0 * running_frac_pos_triplets / print_freq
            print(f'[{epoch}, {i}] | loss: {avg_loss} | % avg hard triplets: {avg_trip}%')
            running_loss = 0
            running_frac_pos_triplets = 0

    return avg_loss


def find_best_threshold(dists, targets):
    best_thresh = 0.01
    best_correct = 0
    for thresh in torch.arange(0.0, 1.51, 0.01):
        predictions = dists <= thresh.cuda()
        correct = torch.sum(predictions == targets.cuda()).item()
        if correct > best_correct:
            best_thresh = thresh
            best_correct = correct

    accuracy = 100.0 * best_correct / dists.size(0)

    return best_thresh, accuracy


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    embeds, labels = [], []

    for data in loader:
        samples, _labels = data[0].cuda(), data[1]
        out = model.get_embedding(samples)
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    dists = torch.cdist(embeds, embeds)

    labels = labels.unsqueeze(0)
    targets = labels == labels.t()

    mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
    dists = dists[mask == 1]
    targets = targets[mask == 1]

    threshold, accuracy = find_best_threshold(dists, targets)

    print('accuracy: {:.3f}%, threshold: {:.2f}'.format(accuracy, threshold))

    return accuracy


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.ColorJitter(0.5, 0.5, 0.5, 0)),
        transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def train(output_path: str,
          epochs: int,
          lr: float,
          p: int,
          k: int,
          margin: float,
          num_workers: int) -> NoReturn:

    train_ml_database_path = Path.joinpath(Path(__file__).parent, '../../data/ml_database/train')
    test_ml_database_path = Path.joinpath(Path(__file__).parent, '../../data/ml_database/test')

    train_dataset = ImageFolder(root=train_ml_database_path, transform=get_transform(train=True))
    train_sampler = PKSampler(train_dataset.targets, p=p, k=k)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=p*k,
                              sampler=train_sampler,
                              num_workers=num_workers)

    test_dataset = ImageFolder(root=test_ml_database_path, transform=get_transform(train=False))
    val_loader = DataLoader(dataset=test_dataset,
                            batch_size=p*k,
                            shuffle=False,
                            num_workers=num_workers)

    model = EmbeddingNet(num_dims=128)
    model.cuda()

    criterion = TripletMarginLoss(margin=margin, mining='batch_hard')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, 8, gamma=0.1)

    #writer = SummaryWriter(os.path.join(log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    with open(f'loss_acc_{p}_{k}_{margin}_{epochs}.txt', 'w') as f:
        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs} ...')

            loss = train_epoch(model, optimizer, criterion, train_loader, epoch)

            scheduler.step()

            acc = evaluate(model, val_loader)

            f.write(f"{epoch},{loss},{acc}\n")

            if epoch % 1 == 0:
                figure = plot_embeddings(model, val_loader, f'embeddings_{p}_{k}_{margin}_{epochs}.png')

            os.makedirs(output_path, exist_ok=True)
            torch.save(model, str(Path.joinpath(Path(output_path), f'epoch_{str(epoch)}.pth')))


