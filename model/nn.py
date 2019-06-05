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


class NeuralNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(NeuralNet,self).__init__()
        #expected input size is 299*299*3 = 268203
        self.layer1 = nn.Sequential(
            nn.Linear(268203,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(16,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(32,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.outputLayer = nn.Linear(64,num_classes)

    def forward(self,x):
        input_dim = self.num_flat_features(x)
        x = x.view(-1,input_dim)#flatten
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.outputLayer(out)
        return out

    def num_flat_features(self,x):
        size = x.size()[1:] #all dimensions except thhe number of batches
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def nn_classifier(dataloaders, num_epochs = 25, lr=0.001,  step_size = 5,gamma=0.1):
        # Number of classes in the dataset
        num_classes = 2
        
        # Batch size for training (change depending on how much memory you have)
        batch_size = 4
        
        # Number of epochs to train for
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = NeuralNet(num_classes).to(device)
        print("Params to learn:")
        params_to_update =[]
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params_to_update, lr= lr)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        model, hist = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler,device, num_epochs=num_epochs, is_inception=False)
        return model, hist