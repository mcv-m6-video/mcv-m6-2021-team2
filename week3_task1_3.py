import cv2
import numpy as np
from typing import OrderedDict, NoReturn

import torch
import logging
import os


import os
from collections import defaultdict, OrderedDict
import xml.etree.ElementTree as ET

import torch
import torch.utils.data
import torchvision

import numpy as np
import cv2

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.models.torchvision_engine import evaluate, train_one_epoch
from src.models.torchvision_utils import collate_fn

def parse_annotations(ann_file):
    tree = ET.parse(ann_file)
    root = tree.getroot()

    annotations = defaultdict(list)
    for track in root.findall('track'):
        if track.attrib['label'] == 'car':
            for box in track.findall('box'):
                frame = int(box.attrib['frame'])
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])
                annotations[frame].append([xtl, ytl, xbr, ybr])

    return OrderedDict(annotations)

class AICityDataset(torch.utils.data.Dataset):
    def __init__(self, video_file, ann_file, transform=None):
        self.video_file = video_file
        self.ann_file = ann_file
        self.transform = transform

        self.boxes = parse_annotations(self.ann_file)
        self.video_cap = cv2.VideoCapture(self.video_file)

        self.classes = ['background', 'person', 'bicycle', 'car']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

    def __getitem__(self, idx):
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = self.video_cap.read()

        if self.transform is not None:
            img = self.transform(img)

        boxes = torch.as_tensor(self.boxes.get(idx, []), dtype=torch.float32)
        labels = torch.full((len(boxes),), self.class_to_idx['car'], dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}

        return img, target

    def __len__(self):
        return int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_data_loaders(root):
    # use our dataset and defined transformations
    transform = torchvision.transforms.ToTensor()
    dataset = AICityDataset(video_file=os.path.join(root, './vdo.avi'),
                            ann_file=os.path.join(root, './s03_c010-annotation.xml'),
                            transform=transform)

    # split the dataset in train and test set
    #indices = np.random.permutation(len(dataset))
    indices = range(len(dataset))
    split = int(len(dataset) * 0.25)
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[:split])
    test_sampler = torch.utils.data.SubsetRandomSampler(indices[split:])

    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=train_sampler, num_workers=1,
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=test_sampler, num_workers=1,
                                              collate_fn=collate_fn)

    return train_loader, test_loader

def get_model(architecture='maskrcnn', finetune=True, num_classes=4):
    # load an instance segmentation model pre-trained pre-trained on COCO
    if architecture == 'fasterrcnn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif architecture == 'maskrcnn':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    else:
        raise ValueError('Unknown detection architecture.')

    if finetune:
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # remove mask heads
    model.roi_heads.mask_roi_pool = None
    model.roi_heads.mask_head = None
    model.roi_heads.mask_predictor = None

    return model


def train(model, train_loader, test_loader, device, num_epochs=1, save_path=None):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, test_loader, device, save_path)

np.random.seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_loader, test_loader = get_data_loaders(root='data')
model = get_model(architecture='fasterrcnn', finetune=True, num_classes=len(train_loader.dataset.classes))
model.to(device)

num_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',').__len__()
print(num_gpus)
train(model, train_loader, test_loader, device)
