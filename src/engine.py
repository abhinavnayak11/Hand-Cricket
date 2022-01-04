import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def train(model, trainloader, optimizer, device):
    criterion = nn.CrossEntropyLoss() 
    epoch_loss = []
    epoch_acc = []
    model.train()
    for samples, labels in tqdm(trainloader):
        samples, labels = samples.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # pred = (outputs>0).float()

        pred = torch.argmax(outputs, dim=1)
        correct = pred.eq(labels)
        acc = torch.mean(correct.float())
        
        epoch_loss.append(loss.cpu().item())    # bring to cpu and get the value
        epoch_acc.append(acc.cpu().item())
        
    avg_train_loss = np.mean(epoch_loss)
    avg_train_acc = np.mean(epoch_acc)
        
    print(f'Train Loss: {avg_train_loss:.3f}, Train Accuracy: {avg_train_acc:.3f}', end = " ")

    return avg_train_loss, avg_train_acc

def validate(model, validloader, device):
    criterion = nn.CrossEntropyLoss() 
    epoch_loss = []
    epoch_acc = []

    # model.eval() will notify all your layers that you are in eval mode, that way, 
    # batchnorm or dropout layers will work in eval mode instead of training mode.

    # torch.no_grad() impacts the autograd engine and deactivate it. It will reduce 
    # memory usage and speed up computations but you won’t be able to backprop 
    # (which you don’t want in an eval script).

    model.eval()
    with torch.no_grad():
        for samples, labels in tqdm(validloader):
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            loss = criterion(outputs, labels)

            # pred = (outputs>0).float()   # when using BCELoss

            pred = torch.argmax(outputs, dim=1)
            correct = pred.eq(labels)
            acc = torch.mean(correct.float())
            epoch_loss.append(loss.cpu().item())
            epoch_acc.append(acc.cpu().item())
    
    avg_valid_loss = np.mean(epoch_loss)
    avg_valid_acc = np.mean(epoch_acc)
        
    print(f'| Valid Loss: {avg_valid_loss:.3f}, Valid Accuracy: {avg_valid_acc:.3f}')

    return avg_valid_loss, avg_valid_acc