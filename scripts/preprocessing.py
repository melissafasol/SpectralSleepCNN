import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from torch import nn
from torchvision import datasets, transforms
import torch.optim as optim
import math
import pandas as pd

from torch.utils.data import WeightedRandomSampler, DataLoader

class SleepDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.sleep_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.sleep_frame)

    def __getitem__(self, idx):
        
        time_epoch = self.sleep_frame.iloc[idx]
        y_label = time_epoch['Sleepstage']
        animal_id = time_epoch['Animal_ID']
        epoch_idx = time_epoch['Idx']
        
        img_name = f'{self.root_dir}/{animal_id}_{epoch_idx}_{y_label}.png'
        image = Image.open(img_name).convert('RGB')
    
        if self.transform:
            image = self.transform(image)

        return image, y_label


def make_balanced_sampler(y):
    
    #compute weights for compensating imbalanced classes 
    classes, counts = y.unique(return_counts = True)
    weights = 1.0/counts.float()
    sample_weights = weights[y.squeeze().long()]
    
    #builds sampler with compute weights 
    generator = torch.Generator()
    sampler = WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(sample_weights),
        generator=generator,
        replacement = True
    )
    
    return sampler 

#Composed transform for training data - class from torchvision that allows you to chain together multiple transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

#transform for test data 
normalizer_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

