import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import re

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

import torch
import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

import albumentations as A
import albumentations.pytorch as AP
from albumentations.augmentations.geometric import rotate as AR

from src.dataset import ClassificationDataset


# densenet121

model = torchvision.models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),
    # when using CrossEntropyLoss(), the output from NN should be logit values for each class
    # nn.Linear(500,1)
    # when using BCEWithLogitsLoss(), the output from NN should be logit value for True label
    nn.Linear(500, 7)
)

model.load_state_dict(torch.load('models/hand-cricket-model2.pth', map_location=torch.device('cpu')))


if __name__=='__main__':
    df = pd.read_csv('folds/train.csv')

    images = df['file_path'].values.tolist()
    targets = df['target'].values


    train_transform = A.Compose([
        AR.Rotate(limit = 30),
        A.Resize(128, 128),
        A.HorizontalFlip(),
        A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), max_pixel_value = 255.0, always_apply=True
    )
    ])

    valid_transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), max_pixel_value = 255.0, always_apply=True
    )
    ])

    train_images, valid_images, train_targets, valid_targets = train_test_split(images, 
                                                                                targets, 
                                                                                stratify = targets,
                                                                                random_state = 42)

    train_data = ClassificationDataset(train_images, train_targets, augmentations = train_transform)
    trainloader = DataLoader(train_data, batch_size = 12, shuffle = True, num_workers = 2)

    valid_data = ClassificationDataset(valid_images, valid_targets, augmentations = valid_transform)
    validloader = DataLoader(valid_data, batch_size = 12, shuffle = True, num_workers = 2)

# --------- Training ------- #

    # epoch_acc = []

    # df = pd.DataFrame()
    # for i in range(7):
    #     df[f'{i}'] = [0]*7
    # df.index.name = 'true\predicted'

    # with torch.no_grad():
    #     for samples, labels in tqdm(trainloader):
    #         # samples, labels = samples.to(device), labels.to(device)
    #         outputs = model(samples)

    #         # pred = (outputs>0).float()   # when using BCELoss

    #         preds = torch.argmax(outputs, dim=1)
    #         for true, pred in zip(labels, preds):
    #             df.loc[true.item(), f'{pred.item()}'] += 1

    #         correct = preds.eq(labels)
    #         acc = torch.mean(correct.float())
    #         epoch_acc.append(acc.cpu().item())

    # avg_valid_acc = np.mean(epoch_acc)

    # print(f'Training Accuracy: {avg_valid_acc:.3f}')
    # print(df)

# --------- Validation ------- #


    epoch_acc = []

    df = pd.DataFrame()
    for i in range(7):
        df[f'{i}'] = [0]*7
    df.index.name = 'true\predicted'

    with torch.no_grad():
        for samples, labels in tqdm(validloader):
            # samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)

            # pred = (outputs>0).float()   # when using BCELoss

            preds = torch.argmax(outputs, dim=1)
            for true, pred in zip(labels, preds):
                df.loc[true.item(), f'{pred.item()}'] += 1

            correct = preds.eq(labels)
            acc = torch.mean(correct.float())
            epoch_acc.append(acc.cpu().item())

    avg_valid_acc = np.mean(epoch_acc)

    print(f'Valid Accuracy: {avg_valid_acc:.3f}')
    print(df)