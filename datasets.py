import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data  import random_split, Subset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T

import time
import copy

from model.data_loader_pix2pix import *

PHOTOSHOPS_FULL ='data/photoshops_resized'
ORIGINALS_FULL ='data/originals_resized'


def get_dataloaders(direction):
    master_dataset = Pix2PixDataset(ORIGINALS_FULL, PHOTOSHOPS_FULL, direction)
    print("Size of master dataset: {}".format(len(master_dataset)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    n = len(master_dataset)
    n_test= int( n * .15 )  # number of test/val elements
    n_val = n_test
    n_train = n - 2 * n_test
    train_set, val_set, test_set = random_split(master_dataset, (n_train, n_val, n_test))   
    train_set.transform = data_transforms['train']    
    val_set.transform = data_transforms['val']
    test_set.transform = data_transforms['val']
    
    
    n_train_dev = int( n_train * .25 )
    n_test_dev= int( n_test * .25 )  
    n_val_dev = int( n_val * .25)

    
    train_set_dev = Subset(train_set, range(n_train_dev))
    val_set_dev = Subset(val_set,range(n_val_dev))
    test_set_dev = Subset(test_set, range(n_test_dev))
    print("Size of subsets:\nTrain Dev:{}\tVal Dev:{}\tTest Dev:{}".format(len(train_set_dev), len(val_set_dev), len(test_set_dev)))
    full_dataloaders = {
        'train' : DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, drop_last = True),
        'val' : DataLoader(val_set, batch_size=128, shuffle=True, num_workers=2, drop_last = True),
        'test' : DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2, drop_last = True),
    }
    
    dev_dataloaders = {
        'train' : DataLoader(train_set_dev, batch_size=8, shuffle=True, num_workers=2, drop_last = True),
        'val' : DataLoader(val_set_dev, batch_size=8, shuffle=True, num_workers=2, drop_last = True),
        'test' : DataLoader(test_set_dev, batch_size=8, shuffle=True, num_workers=2, drop_last = True)
    }
    return dev_dataloaders['train'],dev_dataloaders['test'], dev_dataloaders['val']
