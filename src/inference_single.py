import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import re

import torch
import torch.nn as nn

import torch
import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import tqdm
from PIL import Image

import albumentations as A

from src.dataset import ClassificationDataset
from src.display import display_inference_result

# resnet18

# model = torchvision.models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 500),
#     # when using CrossEntropyLoss(), the output from NN should be logit values for each class
#     # nn.Linear(500,1)
#     # when using BCEWithLogitsLoss(), the output from NN should be logit value for True label
#     nn.Linear(500, 7)
# )

# densetnet121

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
model.eval() 

files = ['input/0_9.jpg', 'input/1_50.jpg', 'input/2_83.jpg', 'input/3_100.jpg', 'input/4_140.jpg']
files = glob.glob('/home/abhinavnayak11/Pictures/Webcam/*')  # 1, 2

if __name__=='__main__':

    
    valid_transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), max_pixel_value = 255.0, always_apply=True
    )
    ])

    # valid_images = ['test-images/depositphotos_2209844-stock-photo-hand-is-counting-number-2.jpg']
    valid_images = files
    batch = len(valid_images)
    valid_targets = [-1]*batch

    valid_data = ClassificationDataset(valid_images, valid_targets, augmentations = valid_transform)
    validloader = DataLoader(valid_data, batch_size = batch, shuffle = True, num_workers = 2)

    with torch.no_grad():
        for samples, targets in validloader:
            outputs = model(samples)
            predictions = torch.argmax(outputs, dim=1)

    display_inference_result(samples, predictions, outputs, denorm = True)

