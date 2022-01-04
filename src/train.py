import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import tqdm
from PIL import Image

import albumentations as A
import albumentations.pytorch as AP
from albumentations.augmentations.geometric import rotate as AR

from src.dataset import ClassificationDataset
from src.display import display_images
from src.model import get_model
from src import engine

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
    print(len(train_images), len(valid_images), len(train_targets), len(valid_targets))

    train_data = ClassificationDataset(train_images, train_targets, augmentations = train_transform)
    trainloader = DataLoader(train_data, batch_size = 12, shuffle = True, num_workers = 2)
    # display_images(trainloader, 'training images', batch_size = 12, denorm = True)

    valid_data = ClassificationDataset(valid_images, valid_targets, augmentations = valid_transform)
    validloader = DataLoader(valid_data, batch_size = 12, shuffle = True, num_workers = 2)
    # display_images(validloader, 'valid images', batch_size = 12, denorm = True)

    # fetch model
    model = get_model()
    device = torch.device('cpu')
    model = model.to(device)  

    # define optimizer  
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    # train and validate the model
    train_losses = []
    valid_losses = []
    min_loss = np.inf
    epochs = 2

    for epoch in range(epochs):
        print(f'[Epoch {epoch+1}/{epochs}]:')
        train_loss, _ = engine.train(model, trainloader, optimizer, device)
        valid_loss, _ = engine.validate(model, validloader, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if (valid_loss<min_loss):
            print("Loss decreased. Saving model...")
            min_loss = valid_loss
            torch.save(model.state_dict(), 'models/hand-cricket_model.pth')

    # plot the training results
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend()
    plt.title('Training & Validation loss')
    plt.show()