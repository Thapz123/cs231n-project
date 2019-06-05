import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data  import random_split, Subset
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
import copy
import shutil
TEST_SUITE_PATH = './tests/'

from model.data_loader import *
def test_suite(model_init, params, dataloaders):
    ss, g, lr = params
    model, hist = model_init(dataloaders, num_epochs = 25, lr=lr,  step_size = ss,gamma=g)
    model_wts = copy.deepcopy(model.state_dict())
    acc = np.amax(hist)
    info = (model_wts, acc, hist, params)
    return info
    

def get_data():
    PHOTOSHOPS_FULL ='data/photoshops_resized'
    ORIGINALS_FULL ='data/originals_resized'

    master_dataset = PhotoshopDataset(ORIGINALS_FULL, PHOTOSHOPS_FULL)
    
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
    
    
    n_train_dev = int( n_train * .5 )
    n_test_dev= int( n_test * .5 )  
    n_val_dev = int( n_val * .5)

    
    train_set_dev = Subset(train_set, range(n_train_dev))
    val_set_dev = Subset(val_set,range(n_val_dev))
    test_set_dev = Subset(test_set, range(n_test_dev))
    print("Size of subsets:\nTrain Dev:{}\tVal Dev:{}\tTest Dev:{}".format(len(train_set_dev), len(val_set_dev), len(test_set_dev)))
    full_dataloaders = {
        'train' : DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, drop_last = True),
        'val' : DataLoader(val_set, batch_size=64, shuffle=True, num_workers=2, drop_last = True),
        'test' : DataLoader(test_set, batch_size=64, shuffle=True, num_workers=2, drop_last = True),
    }
    
    dev_dataloaders = {
        'train' : DataLoader(train_set_dev, batch_size=64, shuffle=True, num_workers=2, drop_last = True),
        'val' : DataLoader(val_set_dev, batch_size=64, shuffle=True, num_workers=2, drop_last = True),
        'test' : DataLoader(test_set_dev, batch_size=64, shuffle=True, num_workers=2, drop_last = True)
    }
    return full_dataloaders

#Importing Inception Classifier
from model.inception_net import inception_classifier
from model.cnn import cnn_classifier
from model.nn import nn_classifier

import json

classifiers = [inception_classifier, cnn_classifier, nn_classifier]
classifier_names = ['inception_classifier', 'cnn_classifier', 'nn_classifier']
# These are the best hyperparams 
params = [(5,  0.1, 0.001),(2,  0.05, 0.01),(2,  0.1, 0.005) ]
dataloaders = get_data()
for i, name in enumerate(classifier_names):
    print("Started Training on {}\n".format(name))
    best_model_wts, best_acc, best_hist, best_params =test_suite(classifiers[i], params[i], dataloaders)
    if not(os.path.exists(TEST_SUITE_PATH)):
        os.mkdir(TEST_SUITE_PATH)
    print("dir is {}".format(TEST_SUITE_PATH))
    torch.save(best_model_wts, "{}{}.pt".format(TEST_SUITE_PATH, name))
    best_hist = [acc.data.cpu().numpy().tolist() for acc in best_hist]
    dic = {'acc':best_acc.data.cpu().numpy().tolist(), 'hist':best_hist, 'params': best_params}
    with open("{}{}.json".format(TEST_SUITE_PATH, name), 'w') as json_file:  
        json.dump(dic, json_file)
    print("Completed full training on {}".format(name))


    
    

