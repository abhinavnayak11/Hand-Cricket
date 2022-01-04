import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, file_paths, targets, augmentations = None):
        self.files = file_paths
        self.targets = targets  
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        target = self.targets[idx]
        
        # convert to numpy array
        img = np.array(img)
        
        # transform img using albumentations 
        # alb requires input to be numpy array
        if (self.augmentations):
            augmented = self.augmentations(image=img)
            img = augmented["image"]
        
        # pytorch requires image in format (#channels, H, W).
        # currently image is in form (H,W,C)
        img = img.transpose(2,0,1).astype(np.float32)
                
        return (torch.tensor(img, dtype=torch.float), 
                torch.tensor(target, dtype=torch.long))


