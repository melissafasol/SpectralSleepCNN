import os
import torch
from torch import nn
import torch.optim as optim
import math
import pandas as pd


class CNNSleep(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculating the size of the output from the last MaxPool2d layer
        # Assuming the input images are 256x256, after two pooling layers with kernel_size=2, stride=2:
        # Output dimension: (256 / 2) / 2 = 64
        # Output size for each feature map is 64x64, and there are 64 feature maps
        self.flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Linear(64 * 64 * 64, 512),  # Size adjusted to match flattened output
            nn.ReLU(),
            nn.Linear(512, 10),  # Assuming 10 classes for output
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.fc_stack(x)
        return logits
