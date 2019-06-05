from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from .helper_funcs import train_model
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
#Convolutional neural network (two convolutional layers)
class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            
        )
        self.flatten = Flatten()
        self.print = Print()
        self.fc = nn.Linear(1401856, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.flatten(out)
        #out = self.print(out)
        out = self.fc(out)
        return out



            

def cnn_classifier(dataloaders, num_epochs = 25, lr=0.001,  step_size = 5,gamma=0.1):
    # Number of classes in the dataset
    num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    batch_size = 4

    # Number of epochs to train for
  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvNet(num_classes).to(device)
    print("Params to learn:")
    params_to_update =[]
    for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
                
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    model, hist = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler,device, num_epochs=num_epochs, is_inception=False)
    return model, hist
