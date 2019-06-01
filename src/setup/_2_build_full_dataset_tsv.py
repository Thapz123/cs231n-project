from __future__ import print_function, division
import os
import random
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


ORIGINALS_TSV = 'data/originals.tsv'
PHOTOSHOPS_TSV = 'data/photoshops.tsv'
ORIGINALS_FULL_TSV = 'data/originals_full_unprocessed.tsv'
PHOTOSHOPS_FULL_TSV = 'data/photoshops_full_unprocessed.tsv'

# class FaceLandmarksDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, csv_file, root_dir = '../data', transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file, sep = '\t')
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.landmarks_frame)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample


def build_full_datasets_df():
    random.seed(42)
    originals, photoshops = read_tsv_files()
    o_small = originals #.sample(n = 3000, random_state = 42)
    p_small = photoshops[photoshops['original'].isin(o_small['id'])]
    
    #sampling one photoshopped image per real image
    size = 1        # sample size
    replace = False  # with replacement
    fn = lambda obj: obj.sample(size, random_state =42, replace = replace)
    grouped = p_small.groupby('original', axis = 0, as_index = False).apply(fn)
    p_small = pd.DataFrame(grouped)
    return (o_small, p_small)
    

def read_tsv_files():
    or_df = pd.read_csv(ORIGINALS_TSV, sep='\t')
    ph_df = pd.read_csv(PHOTOSHOPS_TSV, sep = '\t')
    or_df = or_df[or_df['end'] == 'jpg']
    ph_df = ph_df[ph_df['end'] == 'jpg']
    return (or_df, ph_df)

def print_dataframes_to_tsv():
    (originals_full, photoshops_full) = build_full_datasets_df()
    originals_full.to_csv(ORIGINALS_FULL_TSV, sep = '\t', index=False)
    photoshops_full.to_csv(PHOTOSHOPS_FULL_TSV, sep = '\t', index=False)
    
def build_full_datasets():
    exists_orig = os.path.isfile(ORIGINALS_FULL_TSV)
    exists_pho = os.path.isfile(PHOTOSHOPS_FULL_TSV)
    if(not(exists_orig and exists_pho)):
        print_dataframes_to_tsv()

def clean_dataset_file(df, root_path, is_photoshop):
    
    ids = df['id']
    if is_photoshop:
        path = os.path.join(root_path, df.where(df[id] == id)['original'],id)
        ids = [id for id in ids if os.path.isfile(path)]
    else:
        path = os.path.join(root_path,id)
        ids = [id for id in ids if os.path.isfile(path)]
    clean_df = df[df['id'].isin(ids)]
    return clean_df


    
    
    